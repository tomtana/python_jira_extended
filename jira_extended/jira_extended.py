import collections
from functools import lru_cache, reduce
from operator import getitem
import pandas as pd
from jira import JIRA
from jira.client import translate_resource_args, ResultList
from jira.resources import GreenHopperResource, Comment, Issue
import copy
import numpy as np
import json
import re
from box import Box
import logging


class JIRA_EXT(JIRA):
    def __init__(self, *args, logger=logging.getLogger('jira_ext'), **kwargs):
        super(JIRA_EXT, self).__init__(*args, **kwargs)
        self._init_class_properties(logger)

    def _init_class_properties(self, logger=logging.getLogger('jira_ext')):
        """
        dirty workaround since child class will need to init the added properties but no super.__init__ can be called
        """
        self.__logger = logger
        self.__issue_cache = collections.OrderedDict()
        self._field_mappings = collections.defaultdict(lambda: collections.defaultdict(lambda: None))

    @translate_resource_args
    def sprint_scope_and_burndown_chart(self, board_id, sprint_id):
        data = {}
        data['rapidViewId'] = board_id
        data['sprintId'] = sprint_id

        if self._options['agile_rest_path'] == GreenHopperResource.GREENHOPPER_REST_PATH:
            # Old, private API did not support pagination, all records were present in response,
            #   and no parameters were supported.
            r_json = self._get_json('rapid/charts/scopechangeburndownchart', params=data, base=self.AGILE_BASE_URL)
            # boards = [Board(self._options, self._session, raw_boards_json) for raw_boards_json in r_json['views']]
            # return ResultList(boards, 0, len(boards), len(boards), True)
            return r_json
        else:
            # return self._fetch_pages(Board, 'values', 'board', startAt, maxResults, params, base=self.AGILE_BASE_URL)
            print("Not implemented for this API Version!!")
            pass

    def sprint_report(self, board_id, sprint_id):
        """Return the total incompleted points this sprint."""
        return self._get_json('rapid/charts/sprintreport?rapidViewId=%s&sprintId=%s' % (board_id, sprint_id),
                              base=self.AGILE_BASE_URL)

    def get_linked_issues(self, issue, link_type="is precondition for", regex=".*", generic_filter=None,
                          load_linked_issues=True):
        '''
            Returns all issues linked to issue which: 
                * have a link matching the regex type=x
                * linked issue's summary matches regex=x
        '''
        import re
        if not issue:
            print("issues is empty")
        result = ResultListExt()

        links = issue.fields.issuelinks
        # if they have already linked features check if the naming is okay and all the defined features are there
        # print(i.fields.summary)
        # feature_search_str=".*"+get_epic_full_name(i.fields.summary)
        # print(links)
        if links:
            for idx, l in enumerate(links):
                if hasattr(l, 'outwardIssue') and hasattr(l.type, 'outward'):
                    # print('outward: '+ l.type.outward+" "+l.outwardIssue.fields.summary)
                    # check if link is of right type
                    if re.search(link_type, l.type.outward):
                        # check linked issue name
                        if re.search(regex, l.outwardIssue.fields.summary, re.IGNORECASE):
                            # print('Outward: Feature found'+ i.fields.summary)
                            result.append(l.outwardIssue)

                if hasattr(l, 'inwardIssue') and hasattr(l.type, 'inward'):
                    # print('inward: '+ l.type.inward+" "+l.inwardIssue.fields.summary)
                    # check if link is of right type
                    if re.search(link_type, l.type.inward):
                        # check linked issue name
                        if re.search(regex, l.inwardIssue.fields.summary):
                            # print('Inward: Feature found'+ i.fields.summary)
                            result.append(l.inwardIssue)

        if result:
            cached = self.get_from_issue_cache(result)
            uncached = self.get_uncached_issues(result)
            if load_linked_issues and uncached:
                result = self.search_issues(self.create_jql_for_issuelist(uncached))
                self.update_issue_cache(result)
                result += cached
            else:
                dict_result = dict(np.array([[i.key for i in result], result]).transpose())
                result = [dict_result.get(i) for i in uncached]
                result += cached

            if generic_filter:
                result = [i for i in filter(generic_filter, result)]

        return result

    def search_issue_chunks_with_issuekeys(self, issue_keys, chunk_size=None):
        result = ResultListExt()
        if issue_keys is None:
            raise IOError("A list of issue_keys needs to be passed!")

        if issue_keys:
            if chunk_size:
                iks = np.array_split(issue_keys, np.ceil(len(issue_keys) / chunk_size))
                for i in iks:
                    jql = self.create_jql_for_issuelist(i.tolist())
                    result += self.search_issues(jql, maxResults=chunk_size)
            else:
                jql = self.create_jql_for_issuelist(issue_keys)
                result = self.search_issues(jql, maxResults=chunk_size)
            self.update_issue_cache(result)
        return result

    def search_issue_chunks_with_jql(self, jql, chunk_size=None):
        result = ResultListExt()
        if jql is None:
            raise IOError("An jql expression needs to be passed!")

        if not isinstance(jql, str):
            raise IOError("An string needs to be passed")

        if jql:
            if chunk_size:
                num_curr = chunk_size
                cnt = 0
                while chunk_size == num_curr:
                    i = self.search_issues(jql, startAt=cnt, maxResults=chunk_size)
                    result += i
                    num_curr = len(i)
                    cnt += len(i)
            else:
                result = self.search_issues(jql, maxResults=chunk_size)
            self.update_issue_cache(result)
        return result

    @staticmethod
    def parse_greenhopper_sprints(sprints):
        '''
        Parses the sprint string list and returns a dict with key:value
        '''
        sprints_parsed = []

        # if sprints were assigned parse them
        if sprints:
            for s in sprints:
                s = re.search('\[(.*)\]', s).group(1)
                # str_list = s.split(',')
                sprint_dict = Box({})
                while True:
                    m = re.match('(^[a-z,A-Z]*=).*?(?=,[a-z,A-Z]*=|$)', s)
                    if not m:
                        break

                    e = re.split('=', m.group())
                    s = s[m.end() + 1::]
                    # print(e)
                    if e[1] == '<null>':
                        e[1] = None
                    sprint_dict[e[0]] = e[1]
                sprints_parsed.append(sprint_dict)
        return sprints_parsed

    def issue_sprints(self, issue):
        '''
        Parses the sprint string list and returns a dict with key:value
        '''
        sprint = []

        # if issue has not the field included load it from server
        if 'customfield_10004' not in issue.raw['fields']:
            issue = self.issue(issue.key)

        return self.parse_greenhopper_sprints(issue.fields["customfield_10004"])

    def issue(self, id, fields=None, expand=None, cache=False, translate_custom_field_name=True):
        """Get an issue Resource from the server.

        :param id: ID or key of the issue to get
        :param fields: comma-separated string of issue fields to include in the results
        :param expand: extra information to fetch inside each resource
        """
        i = super(JIRA_EXT, self).issue(id, fields, expand)
        if translate_custom_field_name:
            i = IssueExt(i, field_mapping=self.get_field_mapping(i), jira_ext=self)
        else:
            i = IssueExt(i, jira_ext=self)
        if cache:
            self.update_issue_cache([i])
        return i

    def search_issues(self, *args, cache_result=True, translate_custom_field_name=True, **kwargs):
        issues = super(JIRA_EXT, self).search_issues(*args, **kwargs)
        if translate_custom_field_name:
            issues = ResultListExt(
                [IssueExt(i, field_mapping=self.get_field_mapping(i), jira_ext=self) for i in issues])
        else:
            issues = ResultListExt([IssueExt(i, jira_ext=self) for i in issues])
        if cache_result:
            self.update_issue_cache(issues)
        return issues

    def get_field_mapping(self, issue):
        if not ("project" in issue.raw["fields"] and "issuetype" in issue.raw["fields"]):
            i = self.issue(issue.key, fields=["project", "issuetype"])
        else:
            i = issue
        if self._field_mappings[i.fields.project.key][i.fields.issuetype.name]:
            return self._field_mappings[i.fields.project.key][i.fields.issuetype.name]
        else:
            meta = self.editmeta(i)["fields"]
            meta_mod = copy.deepcopy(meta)
            for k, v in meta.items():
                sanitized_name = re.sub(r'[*-+/\\.,~\- ]', r'_', v["name"].casefold())
                sanitized_name = re.sub(r'(_){2,}', r'_', sanitized_name)
                meta_mod[k]["sanitized_name"] = sanitized_name
                meta_mod[sanitized_name] = k
            self._field_mappings[i.fields.project.key][i.fields.issuetype.name] = meta_mod
            return meta_mod

    @property
    def issue_cache(self):
        return self.__issue_cache

    def update_issue_cache(self, issues):
        if not issues:
            return

        issues = self._listify(issues)
        d = dict(np.array([[i.key for i in issues], issues]).transpose())
        self.__issue_cache = collections.OrderedDict({**self.__issue_cache, **d})

    def clear_issue_cache(self, target_size=None):
        if target_size:
            for i in range(0, min([target_size, len(self.__issue_cache)])):
                self.__issue_cache.popitem(last=False)
        else:
            self.__issue_cache.clear()

    def get_from_issue_cache(self, key_list):
        if not key_list:
            return ResultListExt()
        key_list = self._listify(key_list)

        if isinstance(key_list[0], Issue):
            key_list = [i.key for i in key_list]

        result_in_cache = ResultListExt()
        for k in key_list:
            if k in self.__issue_cache:
                result_in_cache.append(self.__issue_cache[k])
        return result_in_cache

    def get_uncached_issues(self, key_list):
        if not key_list:
            return ResultListExt()
        key_list = self._listify(key_list)
        if isinstance(key_list[0], Issue):
            key_list = [i.key for i in key_list]
        result = ResultListExt()
        for k in key_list:
            if k not in self.__issue_cache:
                result.append(k)
        return result

    def save_cache_to_file(self, path=r".issue_cache.json"):
        with open(path, 'w') as f:
            cache = [i.raw for i in list(self.issue_cache.values())]
            json.dump(cache, f)

    def load_cache_from_file(self, path=r".issue_cache.json"):
        with open(path, 'r') as f:
            raw = json.load(f)
            issues = [Issue(self._options, self._session, r) for r in raw]
            self.update_issue_cache(issues)

    def save_issues_to_file(self, issue_list, path):
        with open(path, 'w') as f:
            cache = [i.raw for i in issue_list]
            json.dump(cache, f)

    def load_issues_from_file(self, path):
        with open(path, 'r') as f:
            raw = json.load(f)
            return ResultListExt([Issue(self._options, self._session, r) for r in raw])

    def get_uncached_issues_jql_request(self, issue_list):
        issues = self.get_uncached_issues(issue_list)
        return self.create_jql_for_issuelist(issues)

    def create_jql_for_issuelist(self, issuelist):
        jql = ""
        issuelist = self._listify(issuelist)
        if not issuelist:
            return ""
        if isinstance(issuelist[0], Issue):
            jql = [i.key for i in issuelist]
        jql = list(dict.fromkeys(issuelist))
        jql = " OR ".join("issuekey = {0}".format(i) for i in jql)
        return jql

    def create_recursive_issue_list(self, issues, link_type_regex, depth=None, generic_filter=None):
        '''
        Crawls all the issues from the given list of issues and searches all linked issues
        '''
        issue_list = ResultListExt()

        linked_issues = ResultListExt()
        # get all links of first level
        for i in issues:
            linked_issues += self.get_linked_issues(i, link_type=link_type_regex, load_linked_issues=False)

        # print("{0} {1}:".format(i.fields.summary,link_type_regex))
        if linked_issues:
            # remove duplicates
            linked_issues = list(dict(np.array([[k.id for k in linked_issues], linked_issues]).transpose()).values())
            # load from server all the fields of the linked issue. retrieve them with the created jql query
            cached = self.get_from_issue_cache([i.key for i in linked_issues])
            linked_issues = self.get_uncached_issues([i.key for i in linked_issues])

            linked_issues = self.search_issue_chunks_with_issuekeys(issue_keys=linked_issues, chunk_size=50)
            self.update_issue_cache(linked_issues)
            linked_issues += cached
            issue_list = linked_issues
            if bool(depth):
                depth_loc = depth - 1
                if depth_loc > 0:
                    issue_list += self.create_recursive_issue_list(linked_issues, link_type_regex,
                                                                   depth=depth_loc)
            else:
                issue_list += self.create_recursive_issue_list(linked_issues, link_type_regex, depth=depth)
            # for li in linked_issues:
            # print( "- {0}".format(i.fields.summary))
        if generic_filter:
            issue_list = filter(generic_filter, issue_list)
            issue_list = [i for i in issue_list]
        return issue_list

    def create_issue_graph(self, issues, link_type_regex, depth=None):
        '''
        Crawls all the issues from the given list of issues and searches all linked issues
        '''
        issue_hierarchy = ResultListExt()
        issue_cache = ResultListExt()

        for i in issues:
            li = [li for li in self.get_linked_issues(i, link_type=link_type_regex, load_linked_issues=False)]
            issue_cache += li
            issue_dict = {"issue": i, "linked_issues": li}
            issue_hierarchy.append(issue_dict)

        if issue_cache:
            cached = self.get_from_issue_cache(issue_cache)
            issue_cache = self.get_uncached_issues(issue_cache)

            issue_cache = self.search_issue_chunks_with_issuekeys(issue_keys=issue_cache, chunk_size=100)
            self.update_issue_cache(issue_cache)
            issue_cache += cached

            issue_cache = dict(np.array([[k.key for k in issue_cache], issue_cache]).transpose())

        for d in issue_hierarchy:

            linked_issues = [issue_cache[i.key] for i in d["linked_issues"]]
            # print("{0} {1}:".format(i.fields.summary,link_type_regex))
            if linked_issues:

                d["linked_issues"] = linked_issues
                if bool(depth):
                    depth_loc = depth - 1
                    if depth_loc > 0:
                        d["linked_issues"] = self.create_issue_graph(linked_issues, link_type_regex,
                                                                     depth=depth_loc)
                else:
                    d["linked_issues"] = self.create_issue_graph(linked_issues, link_type_regex, depth=depth)
                # for li in linked_issues:
                # print( "- {0}".format(i.fields.summary))
        return issue_hierarchy

    def _delistify(self, object):
        if isinstance(object, list):
            if np.size(object) == 1:
                return object[0]
        return object

    def _listify(self, object):
        '''
        Return a list of the passed object
        '''
        if isinstance(object, list):
            return object
        elif isinstance(object, type(np.array([]))):
            return object.tolist()
        elif not object:
            return []
        else:
            return [object]

    def move_to_sprint(self, issues, sprint, recursion_association=None, generic_filter=None, simulate=False):
        '''
        Move all issue with recursion relation to given sprint. Generic filter is a function which allows to filter
        the result of the recursion individually.

        Parameters:
        issues (list): a list of jira issues
        sprint (int): sprint id to which issues should be moved
        recursion_association (string): regular expression to follow linked relation (e.g. "Is mother of")
        generic_filer (func): function to filter recursion results before issues are moved

        Example usage:
        move_to_sprint(issues_ppc,j.get_sprint("11"),"Is mother of",lambda x: not bool(j.issue_curr_sprint(x)))
        '''
        issues_to_move = copy.copy(self._listify(issues))

        if recursion_association:
            issues_to_move += self.create_recursive_issue_list(issues_to_move, recursion_association)

        if generic_filter:
            f = filter(generic_filter, issues_to_move)
            issues_to_move = [i for i in f]

        self.__logger.info(f"The following issues were identified {self.sprint(sprint).name}:\n")
        [print(f"{i.key}\t{i.fields.summary}") for i in issues_to_move]

        if issues_to_move and not simulate:
            self.__logger.info(f"Moving now the issues")
            issues_to_move = [i.key for i in issues_to_move]
            self.add_issues_to_sprint(sprint, issues_to_move)
        else:
            self.__logger.info(f"Simulation active, nothing executed")

    @lru_cache(maxsize=32)
    def get_boards_by_name(self, board_regex):
        f = filter(lambda x, _board_regex=board_regex: re.search(_board_regex, x.name), self.boards())
        return [i for i in f]

    @lru_cache(maxsize=32)
    def get_all_sprints_from_boards(self, board_regex="(AR.*HUD|Audi HCP3 SAFe.*Capability)"):
        boards = self.get_boards_by_name(board_regex)
        sprints = []
        for b in boards:
            sprints += self.sprints(b.id)
        # remove duplicates
        sprints = list(dict(np.array([[k.id for k in sprints], sprints]).transpose()).values())
        return sprints

    @lru_cache(maxsize=32)
    def get_sprints_by_name(self, sprint_regex, board_regex="(AR.*HUD|Audi HCP3 SAFe.*Capability)"):
        sprints = self.get_all_sprints_from_boards(board_regex)
        f = filter(lambda x, _sprint_regex=sprint_regex: re.search(_sprint_regex, x.name), sprints)
        return [sprint for sprint in f]

    def get_sprint(self, pi, sprint="", team="(AR.*HUD)"):
        search_string = ""
        if not sprint:
            search_string = f"AHCP3.*(PI)*{pi}"
        else:
            search_string = f"(PI)*{pi}.*(S)*[0-9]{{0,1}}{sprint}.*{team}"

        self.__logger.debug(f"Search string: {search_string}")
        # print(search_string)
        sprints = self.get_sprints_by_name(search_string)

        if len(sprints) != 1:
            sprints = None
        return self._delistify(sprints)

    def comments(self, issue):
        return CommentList([CommentExt(i) for i in super(JIRA_EXT, self).comments(issue)])

    def comment(self, issue, comment):
        return CommentExt(super(JIRA_EXT, self).comment(issue, comment))


class IssueExt(Issue):
    """
    Convenience class for GCIP issues.
    +   print function shows also summary
    +   customfield names are translated to sanitized given name (works only with GCIPJira)
    """

    def __init__(self, issue=None, field_mapping=None, jira_ext=None, *args, **kwargs):
        if isinstance(issue, Issue):
            kwargs = {'options': issue._options,
                      'raw': issue.raw,
                      'session': issue._session}
        if field_mapping:
            self.__field_mapping = collections.defaultdict(lambda: None, field_mapping)
        else:
            self.__field_mapping = collections.defaultdict(lambda: None)
        super(IssueExt, self).__init__(*args, **kwargs)
        self._parse_raw(self.raw)
        self.__jira = jira_ext

    def __str__(self):
        return f"<{self.key}>\t\t<{self.fields.summary}>"

    def __repr__(self):
        return f"<{self.key}>\t\t<{self.fields.summary}>"

    def _parse_raw(self, raw):
        # add custom fieldnames to the dict if they exist
        raw["clear2custom"] = collections.defaultdict(lambda: None)
        for k, v in self.__field_mapping.items():
            if "fields" in raw:
                if k in raw["fields"] and k.startswith("customfield_"):
                    sanitized_name = v["sanitized_name"]
                    raw["fields"][sanitized_name] = raw["fields"][k]
                    raw["clear2custom"][sanitized_name] = k

        super(IssueExt, self)._parse_raw(raw)

    @property
    def sprints(self):
        ret = {}
        # if issue has not the field included load it from server
        if 'customfield_10004' not in self.raw['fields']:
            self.update()
        return JIRA_EXT.parse_greenhopper_sprints(self.fields.customfield_10004)

    @property
    def current_sprint(self):
        if self.sprints:
            return self.sprints[-1]
        return None

    def to_df(self):
        df = pd.DataFrame(self.raw["fields"])
        for k, v in self.raw.items():
            if k != "fields":
                df[k] = v
        return df

    def move_to_sprint(self, sprint):
        """
        Move Feature to given sprint and update issue fields
        @param sprint: id
        @return: None
        """
        self.__jira.add_issues_to_sprint(sprint, [self.key])
        self.update()

    @property
    def comments(self):
        return self.__jira.comments(self.key)

    def add_comment(self, body, visibility=None, is_internal=False):
        """
        Add a comment from the current authenticated user on the specified issue and return a Resource for it.
        The issue identifier and comment body are required.
        :param issue: ID or key of the issue to add the comment to
        :param body: Text of the comment to add
        :param visibility: a dict containing two entries: "type" and "value".
            "type" is 'role' (or 'group' if the JIRA server has configured
            comment visibility for groups) and 'value' is the name of the role
            (or group) to which viewing of this comment will be restricted.
        :param is_internal: defines whether a comment has to be marked as 'Internal' in Jira Service Desk
        Type:      method
        """
        self.__jira.add_comment(self.key, body, visibility=visibility, is_internal=is_internal)


class ResultListExt(ResultList):
    def __init__(self, iterable=None, mapping=None, **kwargs):
        super(ResultListExt, self).__init__(iterable, **kwargs)
        self.mapping = mapping

    def __getitem__(self, index):
        retval = super(ResultListExt, self).__getitem__(index)
        retval = type(self)(retval)
        return retval

    def raw(self):
        """
        export raw content of each issue
        """
        return [i.raw for i in self]

    def raw_flattened(self):
        return [self.flatten_fields(i.raw) for i in self]

    @staticmethod
    def flatten_fields(raw):
        ret_dict = {**raw.get("fields")}
        for key, value in raw.items():
            if key != "fields":
                ret_dict[key] = value
        return ret_dict

    def normalized_json(self, _mapping=None):
        """
        Export normalized json. If mapping is explicitly given, self.mapping property is ignored

        mapping must have structure:
        "normalized_name": "name",
        "function": function,
        """

        def getitem_from_dict(dataDict, mapList):
            """Iterate nested dictionary"""
            return reduce(getitem, mapList, dataDict)

        mapping = self.mapping
        if _mapping:
            mapping = _mapping
        if not mapping:
            return self.raw_flattened()

        normalized_list = []
        for i in self:
            d = {}
            i_flat_raw = self.flatten_fields(i.raw)
            for key, properties in mapping:
                field = properties.get("field")
                normalized_name = properties.get("normalized_name", key)
                fun = properties.get("function")
                field_content = getitem_from_dict(i_flat_raw, field.split(",./"))
                if field and fun:
                    d[normalized_name] = fun(field_content)
                elif field and not fun:
                    d[normalized_name] = field_content
                elif not field and fun:
                    d[normalized_name] = fun(i)
                else:
                    raise ValueError("Mapping input doesnt contain field nor function")
            normalized_list.append(d)
        return normalized_list

    def normalized_df(self, _mapping=None):
        normalized_json = self.normalized_json(_mapping=_mapping)
        return pd.DataFrame(normalized_json)


class CommentExt(Comment):
    def __init__(self, comment):
        super(CommentExt, self).__init__(options=comment._options, session=comment._session, raw=comment.raw)

    def __str__(self):
        return f"From: {self.author.displayName} {self.updated}\n{self.body}\n____END____"

    def __repr__(self):
        return f"From: {self.author.displayName} {self.updated}\n{self.body}\n____END____\n"


class CommentList(list):
    def __setitem__(self, key, value):
        comment = self[key]
        comment.update(body=value)

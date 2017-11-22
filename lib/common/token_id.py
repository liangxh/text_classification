# -*- coding: utf-8 -*-
from .exceptions import UnknownTokenNotSupportedException, PreservedTokenEncounteredException


class TokenIdMapping(object):
    TOKEN_UNKNOWN = '<UNKNOWN>'

    def __init__(self, token_list, support_unknown=False):
        self.token_to_id = self._build_token_id_mappping(token_list)
        self.support_unknown = support_unknown

        if self.support_unknown:
            self._check_unknown_token_support()
            self.unknown_token_id = self.token_to_id[TOKEN_UNKNOWN]

    def _check_unknown_token_support(self):
        """
        檢查映射表中是否有 '<UNKNOWN>'
        """
        if TOKEN_UNKNOWN not in self.token_to_id:
            raise UnknownTokenNotSupportedException

    def get(self, token):
        """
        返回token對應id
        若support_unknown則返回unknown_token_id, 否則返回None
        """

        if self.support_unknown:
            if token == TOKEN_UNKNOWN:
                raise PreservedTokenEncounteredException

            return self.token_to_id.get(token, self.unknown_token_id)
        else:
            return self.token_to_id.get(token, None)

    def token_list_to_id_list(self, token_list):
        """
        將token列表轉換為id列表
        若非support_unknown則無視不支持的
        """
        if self.support_unknown:
            return [self.get(token) for token in token_list]
        else:
            id_list = list()
            for token in token_list:
                id = self.get(token)
                if id is not None:
                    id_list.append(id)
            return id_list

    @classmethod
    def _build_token_id_mapping(cls, token_list):
        token_to_id = dict()
        for i, token in enumerate(token_list):
            token_to_id[token] = i
        return token_to_id

    @staticmethod
    def load(filename):
        """
        從文件中讀取詞嵌入向量
        """
        token_list = list()
        with open(filename, 'r') as fobj:
            for line in fobj:
                line = line.strip()
                if line == '':
                    continue
                token_list.append(line)
        return TokenIdMapping(token_list)

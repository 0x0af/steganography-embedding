#!/usr/bin/python

# coding: utf-8

"""
Created on Tue Oct 20 2015

@author: Anton at 0x0af@ukr.net
"""

import numpy
from numpy import *
from pseudorandomness_source import *

__all__ = ["HuffmanEncoder", "HuffmanDecoder"]


class HuffmanConfig(PseudorandomnessSourceConfig):
    HUFFMAN = 'Huffman'

    AUX_CODE_MAP = 'huffman_code_map'


class HuffmanNode(object):
    recurPrint = False

    def __init__(self, value=None, fq=None, lnode=None, rnode=None, parent=None):
        self.L = lnode
        self.R = rnode
        self.p = parent
        self.v = value
        self.fq = fq

    def __repr__(self):
        if HuffmanNode.recurPrint:
            lnode = self.L if self.L else '#'
            rnode = self.R if self.R else '#'
            return ''.join(('(%s:%d)' % (self.v, self.fq), str(lnode), str(rnode)))
        else:
            return '(%s:%d)' % (self.v, self.fq)

    def __cmp__(self, other):
        if not isinstance(other, HuffmanNode):
            return super(HuffmanNode, self).__cmp__(other)
        return cmp(self.fq, other.fq)


class HuffmanStack(object):
    @staticmethod
    def _pop_first_two_nodes(nodes):
        if len(nodes) > 1:
            first = nodes.pop(0)
            second = nodes.pop(0)
            return first, second
        else:
            # print "[popFirstTwoNodes] nodes's length <= 1"
            return nodes[0], None

    @staticmethod
    def build_tree(nodes):
        nodes.sort()
        while True:
            first, second = HuffmanStack._pop_first_two_nodes(nodes)
            if not second:
                return first
            parent = HuffmanNode(lnode=first, rnode=second, fq=first.fq + second.fq)
            first.p = parent
            second.p = parent
            nodes.insert(0, parent)
            nodes.sort()

    @staticmethod
    def generate_huffman_code(node, dict_codes, buffer_stack):
        if not node.L and not node.R:
            dict_codes[node.v] = int(''.join(buffer_stack), base=2)
            return
        buffer_stack.append('0')
        HuffmanStack.generate_huffman_code(node.L, dict_codes, buffer_stack)
        buffer_stack.pop()

        buffer_stack.append('1')
        HuffmanStack.generate_huffman_code(node.R, dict_codes, buffer_stack)
        buffer_stack.pop()

    @staticmethod
    def calculate_value_frequency(square_matrix):
        from collections import defaultdict
        d = defaultdict(int)
        for (_, _), element in numpy.ndenumerate(square_matrix):
            d[element] += 1
        return d


class HuffmanEncoder(PseudorandomImageEncoder):
    def __init__(self, square_matrix):
        super(HuffmanEncoder, self).__init__(square_matrix, HuffmanConfig.HUFFMAN)

    def _get_tree_root(self):
        d = HuffmanStack.calculate_value_frequency(self.square_matrix)
        return HuffmanStack.build_tree([HuffmanNode(value=value, fq=int(fq)) for value, fq in d.iteritems()])

    def _get_code_map(self):
        a_dict = {}
        HuffmanStack.generate_huffman_code(self.root, a_dict, [])
        return a_dict

    def _encode(self):
        self.root = self._get_tree_root()
        self.code_map = self._get_code_map()
        temp = numpy.zeros(self.square_matrix.shape)
        for (x, y), element in numpy.ndenumerate(self.square_matrix):
            if 255 >= self.code_map[element] >= 0:
                temp[x][y] = self.code_map[element]
            elif self.code_map[element] > 255:
                temp[x][y] = 255
            elif self.code_map[element] < 0:
                temp[x][y] = 0
        self.aux[HuffmanConfig.AUX_CODE_MAP] = self.code_map
        return temp


class HuffmanDecoder(PseudorandomImageDecoder):
    def __init__(self, square_matrix, aux):
        super(HuffmanDecoder, self).__init__(square_matrix, aux, HuffmanConfig.HUFFMAN)

    def _decode(self):
        temp = numpy.zeros(self.square_matrix.shape)
        values = self.aux[HuffmanConfig.AUX_CODE_MAP].values()
        keys = self.aux[HuffmanConfig.AUX_CODE_MAP].keys()
        # print 'Keys: ', keys
        # print 'Values: ', values
        for (x, y), element in numpy.ndenumerate(self.square_matrix):
            index = numpy.where(values == element)
            if index[0].size != 0:
                temp[x][y] = keys[index[0][0]]
        return temp

# print PseudorandomnessSourceTestSuite.test_source(HuffmanEncoder, HuffmanDecoder, 128)

# x256
#  ---
# Test passed
# Encode time: 1.2653778210310993
# Decode time: 6.750315794081058
#  ---

# x512
#  ---
# Test passed
# Encode time: 4.93967529461006
# Decode time: 28.540306404037242
#  ---

# x1024
#  ---
# Test passed
# Encode time: 13.558324619217652
# Decode time: 101.78283557438276
#  ---

# PseudorandomnessSourceTestSuite.test_source_subjective_quality(HuffmanEncoder, HuffmanDecoder)

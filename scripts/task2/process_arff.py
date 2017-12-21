# -*- coding: utf-8 -*-
import re
import commandr


@commandr.command('text')
def text_to_arff(input_filename, output_filename):
    with open(input_filename, 'r') as in_obj, open(output_filename, 'w') as out_obj:
        out_obj.write('@relation \'RELATION\'\n')
        out_obj.write('@attribute content string\n')
        out_obj.write('@data\n')
        for line in in_obj:
            line = line.strip()
            if line == '':
                continue
            out_obj.write(repr(line) + '\n')


@commandr.command('first_text')
def first_text_to_arff(input_filename, output_filename):
    with open(input_filename, 'r') as in_obj, open(output_filename, 'w') as out_obj:
        out_obj.write('@relation \'RELATION\'\n')
        out_obj.write('@attribute content string\n')
        out_obj.write('@data\n')
        for line in in_obj:
            line = line.strip()
            if line == '':
                continue
            part = line.split('\t', 1)[0]
            out_obj.write(repr(part) + '\n')


@commandr.command('vec')
def arff_feat_to_vec(input_filename, output_filename):
    n_attribute = -1  # 第一列為content
    data_encountered = False
    pattern_attribute = re.compile('^(\d+) ([-+]?\d+(?:\.\d+)?)$')

    with open(input_filename, 'r') as in_obj, open(output_filename, 'w') as out_obj:
        for line in in_obj:
            line = line.strip()
            if line == '' or line.startswith('@relation'):
                continue
            elif line.startswith('@attribute'):
                # 統計attribute總數
                if not data_encountered:
                    n_attribute += 1
            elif line.startswith('@data'):
                # @data可能會重复出現
                data_encountered = True
            else:
                attribute_found = False
                parts = ['0'] * n_attribute
                for part in line[1:-1].split(','):
                    res = pattern_attribute.match(part)
                    if res is None:
                        if attribute_found:
                            raise Exception('possible error not fixed: {}'.format(line))
                    else:
                        attribute_found = True
                        idx = int(res.group(1))
                        score_str = res.group(2)
                        parts[int(idx) - 1] = score_str
                out_obj.write('\t'.join(parts) + '\n')


if __name__ == '__main__':
    commandr.Run()

from sys import argv

def extract(source_file, target_file, min_seq_length=5, max_seq_length=60):
    fsw = open(source_file + '.filtered', 'w')
    ftw = open(target_file + '.filtered', 'w')
    with open(source_file) as fs, open(target_file) as ft:
        for source, target in zip(fs, ft):
            num_source_toks = len(source.strip().split())
            num_target_toks = len(target.strip().split())
            if num_source_toks >= min_seq_length and num_source_toks <= max_seq_length \
                    and num_target_toks >= min_seq_length and num_target_toks <= max_seq_length:
                fsw.write(source)
                ftw.write(target)
    fsw.close()
    ftw.close()


if __name__ == '__main__':
    extract(argv[1], argv[2])

import json
import scipy.stats as stats


def load_data(scd_file):
    data = []
    temp = []
    with open(scd_file, 'r', encoding='utf-8') as fp:
        for line_id, line in enumerate(fp.readlines()):
            if line_id % 8 == 0:
                temp.append(line.strip())
            elif line_id % 8 == 1:
                temp.append(line.strip().split())
            elif line_id % 8 == 2:
                temp.append(line.strip())
            elif line_id % 8 == 3:
                temp.append(line.strip().split())
            elif line_id % 8 == 4:
                temp.append(line.strip())
            elif line_id % 8 == 5:
                temp.append(line.strip().split())
            elif line_id % 8 == 6:
                temp.append(float(line.strip()))
            else:
                data.append(temp)
                temp = []
                continue
    return data


def calcu_scd(subword_sememes1, subword_sememes2, compword_sememes):
    subset1 = set(subword_sememes1)
    subset2 = set(subword_sememes2)
    compset = set(compword_sememes)
    if subset1.union(subset2) == compset:
        return 3
    else:
        subword_set = set(subword_sememes1 + subword_sememes2)
        truth1 = sum([sememe in subset1 for sememe in compword_sememes])
        truth2 = sum([sememe in subset2 for sememe in compword_sememes])
        if (subword_set > set(compword_sememes)) and (truth1 or truth2):
            return 2
        elif len(subword_set.intersection(set(compword_sememes))) != 0:
            return 1
        else:
            return 0


def main():
    file_name = 'scd.txt'
    data = load_data(file_name)
    scd_human = []
    scd_model = []
    for l in data:
        model_assign = calcu_scd(l[1], l[3], l[5])
        scd_human.append(l[6])
        scd_model.append(model_assign)
    Apearsonf = stats.pearsonr(scd_human, scd_model)
    Aspearmanf = stats.spearmanr(scd_human, scd_model)

    print("Pearson Coef. between human annotated SCD and our methods:{}".format(Apearsonf[0]))
    print("Spearman Rank Coef. between human annotated SCD and our methods:{}".format(Aspearmanf.correlation))


if __name__ == '__main__':
    main()

import copy
import json
import csv
import os
import traceback

def get_obj_to_write_from_line_obj(line):
    tgt = copy.deepcopy(line)
    try:
        for field in ["g1_user_prompt", "g2_user_prompt"]:
            field_id = field.split("_user")[0]
            if "<|The Start of Assistant A's Conversation with User|>" in tgt[field]:
                # tgt["Assistant A"] = tgt[field].split("<|The End of Assistant A's Conversation with User|>")[0].replace("<|The Start of Assistant A's Conversation with User|>", "")
                # tgt["Assistant B"] = tgt[field].split("<|The End of Assistant A's Conversation with User|>")[1].replace("<|The Start of Assistant B's Conversation with User|>", "").replace("<|The End of Assistant B's Conversation with User|>", "")
                # tgt["Assistant A"] = \
                #     tgt[field].split("[The Start of Assistant A's Answer]")[1].split(
                #         "[The End of Assistant A's Answer]")[0].strip()
                tgt["{}_Assistant A".format(field_id)] = \
                    tgt[field].split("<|The Start of Assistant A's Conversation with User|>")[1].split(
                        "<|The End of Assistant A's Conversation with User|>")[0].strip()
                tgt["{}_Assistant B".format(field_id)] = \
                    tgt[field].split("<|The Start of Assistant B's Conversation with User|>")[1].split(
                        "<|The End of Assistant B's Conversation with User|>")[0].strip()
                tgt['{}_User_Question'.format(field_id)] = "N/A"
                if "[The Start of Reference Answer]" in tgt[field]:
                    tgt["{}_Reference_Answer".format(field_id)] = tgt[field].split("[The Start of Reference Answer]")[1].split(
                        "[The End of Reference Answer]")[0].strip()
                else:
                    tgt['{}_Reference_Answer'.format(field_id)] = "N/A"
            else:
                if "[User Question]" in tgt[field]:
                    tgt['{}_User_Question'.format(field_id)] = tgt[field].split("[User Question]")[1].split("[The Start of Assistant A's Answer]")[0].strip()
                    if "[The Start of Reference Answer]" in tgt[field]:
                        tgt["{}_Reference_Answer".format(field_id)] = tgt[field].split("[The Start of Reference Answer]")[1].split(
                            "[The End of Reference Answer]")[0].strip()
                    else:
                        tgt["{}_Reference_Answer".format(field_id)] = "N/A"
                    tgt["{}_Assistant A".format(field_id)] = tgt[field].split("[The Start of Assistant A's Answer]")[1].split(
                        "[The End of Assistant A's Answer]")[0].strip()
                    tgt["{}_Assistant B".format(field_id)] = tgt[field].split("[The Start of Assistant B's Answer]")[1].split(
                        "[The End of Assistant B's Answer]")[0].strip()
    except:
        traceback.print_exc()
        print(tgt)
        exit()
    return tgt


if __name__ == '__main__':
    comparison_file_path = "/path/to/FastChat/fastchat/llm_judge/data/mt_bench/model_judgment/gpt-4_pair.jsonl"
    result_dict = dict()
    fieldnames = []
    with open(comparison_file_path, "r") as f:
        comparisons = [json.loads(line) for line in f]
        for line in comparisons:
            if len(fieldnames) == 0:
                fieldnames = list(line.keys())
            # pick out the comparison that contains dpo/ppo
            if "ppo" in line["model_1"].lower() or "ppo" in line["model_2"].lower():
                if "dpo" in line["model_1"].lower() or "dpo" in line["model_2"].lower():
                    model_names = [line["model_1"], line["model_2"]]
                    model_names.sort()
                    draw_name = "{}_vs_{}".format(model_names[0], model_names[1])
                    if draw_name not in result_dict:
                        result_dict[draw_name] = []
                    result_dict[draw_name].append(line)
    # export result_dict to csv
    # using dictwriter
    csv_dir_path = "/path/to/FastChat/fastchat/llm_judge/data/mt_bench/model_judgment/gpt-4_pair_ppo_dpo"
    os.makedirs(csv_dir_path, exist_ok=True)
    fieldnames += ["g1_Assistant A", "g1_Assistant B", "g1_User_Question", "g1_Reference_Answer",
                   "g2_Assistant A", "g2_Assistant B", "g2_User_Question", "g2_Reference_Answer"]
    for draw_name in result_dict.keys():
        csv_file_path = os.path.join(csv_dir_path, "{}.csv".format(draw_name))
        with open(csv_file_path, "w", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            comparisons = result_dict[draw_name]
            # writer.writerows(comparisons)
            for line in comparisons:
                tgt = get_obj_to_write_from_line_obj(line)
                writer.writerow(tgt)
        # create another file that only records the winning case that dpo wins ppo
        csv_file_path = os.path.join(csv_dir_path, "{}_dpo_wins.csv".format(draw_name))
        with open(csv_file_path, "w", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            comparisons = result_dict[draw_name]
            for line in comparisons:
                if line["g1_winner"] == "tie" or line["g1_winner"] != line["g2_winner"]:
                    continue
                else:
                    if line["g1_winner"] == "model_1":
                        tgt = get_obj_to_write_from_line_obj(line)
                        writer.writerow(tgt)
                            # writer.writerow(line)


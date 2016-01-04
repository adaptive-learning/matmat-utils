import pandas as pd
import os

data = "data"

answers = pd.read_csv(os.path.join(data, "answers.csv"))
items = pd.read_csv(os.path.join(data, "items.csv"))
# simulators.to_csv(os.path.join(output_dir, "visualizations.csv"))
skills = pd.read_csv(os.path.join(data, "skills.csv"))

answers = answers.join(items, on="item", rsuffix="_item")

selector = "item"
selector2 = "id" if selector =="item" else "skill_lvl_3"

selected_items = items[(items["skill_lvl_2"] == 3) & (items["visualization"] == "counting")][selector2 ].values
# selected_items = items[(items["skill_lvl_2"] == 27)][selector2].unique()
# selected_items = items[(items["skill_lvl_2"] == 27) & (items["visualization"] == "free_answer")]["id"].values

selected_answers = answers[answers[selector].isin(selected_items)]
selected_answers = selected_answers.drop_duplicates(["student", selector])

students_answers_sr = selected_answers.groupby("student").apply(lambda g: g["correct"].mean()).to_dict()
students_answers_count = selected_answers.groupby("student").size().to_dict()
selected_students = sorted(students_answers_count.keys(), reverse=True, key=lambda s: students_answers_count[s] * (-1 if students_answers_sr[s] == 1 else 1))

selected_answers = selected_answers[selected_answers["student"].isin(selected_students)]


output = ""


for s in selected_students:
    for i in selected_items:
        try:
            output += str(selected_answers[(selected_answers["student"] ==s) & (selected_answers[selector] ==i)]["correct"].values[0])
        except IndexError:
            output += "-"
    output += "\r\n"

output = output[:-2]

with open("export.txt", "w") as f:
    f.write(output)

print output
print

# print items[items["id"].isin(selected_items)]
for i, s  in enumerate(skills[skills["id"].isin(selected_items)]["name"].values):
    print i + 1, s
# print selected_items
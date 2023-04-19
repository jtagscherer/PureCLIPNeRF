import os

scenes = {
  "chair": {
    "dataset": "chair",
    "article": "a",
    "name": "chair",
    "source_prompt": "A green chair"
  },
  "hotdog": {
    "dataset": "hotdog",
    "article": "a",
    "name": "hotdog",
    "source_prompt": "A hotdog on a plate"
  },
  "lego": {
    "dataset": "lego",
    "article": "a",
    "name": "excavator",
    "source_prompt": "A yellow lego excavator"
  }
}

queries = [
  {
    "name": "blue",
    "query": "{} blue {}"
  },
  {
    "name": "pretzel",
    "query": "{} {} made out of pretzel; food photography; photo-realistic"
  },
  {
    "name": "expressionism",
    "query": "{} expressionist {} painted by Pablo Picasso; expressionism; art photography"
  }
]

if os.path.exists('logs/low_exp_vit16'):
  raise Exception('Remove default log path (logs/low_exp_vit16) before running this script!')

for scene in scenes.values():
  for query in queries:
    current_query = query['query'].format(scene["article"], scene["name"]).strip()
    current_query = current_query[:1].upper() + current_query[1:]

    # Compute all metrics
    os.system(f'mv logs/{scene["dataset"]}_{query["name"]} logs/low_exp_vit16')
    command = f'python run.py --config configs/low/exp_vit16.py --prompt "{current_query}" --source_prompt "{scene["source_prompt"]}" --dataset datasets/{scene["dataset"]}/ --render_only --render_test'
    print(f'[STARTING] {command}')
    os.system(command)
    os.system(f'mv logs/low_exp_vit16 logs/{scene["dataset"]}_{query["name"]}')

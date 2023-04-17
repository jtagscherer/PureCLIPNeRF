import os

scenes = {
  "chair": {
    "dataset": "chair",
    "article": "a",
    "name": "chair"
  },
  "hotdog": {
    "dataset": "hotdog",
    "article": "a",
    "name": "hotdog"
  },
  "lego": {
    "dataset": "lego",
    "article": "a",
    "name": "excavator"
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

for scene in scenes.values():
  for query in queries:
    current_query = query['query'].format(scene["article"], scene["name"]).strip()
    current_query = current_query[:1].upper() + current_query[1:]

    # Render stylized scene
    command = f'python run.py --config configs/low/exp_vit16.py --prompt "{current_query}" --dataset datasets/{scene["dataset"]}/ --i_print 500 --i_weights 1000 --iters 1000'
    print(f'[STARTING] {command}')
    os.system(command)
    os.system(f'mv logs/low_exp_vit16 logs/{scene["dataset"]}_{query["name"]}')

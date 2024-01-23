import pandas as pd
import numpy as np
from collections import Counter

#The Percentage of Words Known in a Text and Reading Comprehension
#indicates that 95-98% coverage is needed for comprehension

NUM_TAGS = 100

data = pd.read_csv('../data/all_data.csv')

vis_data = data.query('type == "Video"')

tags = Counter()

for tag in vis_data['tags']:
    tag = tag.split(',')
    for t in tag:
        tags.update([t])

top_tags = [t[0] for t in tags.most_common(NUM_TAGS)[1:]]
counts = [t[1] for t in tags.most_common(NUM_TAGS)[1:]]
print(top_tags)

coverage = 0
for t in vis_data['tags']:
    t = t.split(',')
    if any([x in t for x in top_tags]):
        coverage += 1

print(f'Total number of tags found: {len(tags)}')
print(f'Coverage of top tags: {coverage/len(vis_data)}')
print(f'Number of occurances in top tag: {tags.most_common(2)[1]}')
print(f'Number of occurances in last tag: {tags.most_common(NUM_TAGS)[-1]}')
print(f'Median counts of top tags {np.median(counts)}')
print(f'Number of occurances of "other" tag: {len(vis_data) - coverage}')

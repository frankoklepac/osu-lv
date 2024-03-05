from collections import defaultdict

word_count = defaultdict(int)

with open('LV1\song.txt', 'r') as f:
    for line in f:
        words = line.split()
        for word in words:
            word_count[word] += 1

single_occurance = [word for word, count in word_count.items() if count == 1]
print("Words that appear only once:", single_occurance)
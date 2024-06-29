import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='10M', help="10M or 100M")
args = parser.parse_args()

input_path = f"../data/processed_{args.dataset}/all.txt"
output_path = f"../data/processed_{args.dataset}/all_par.txt"

with open(output_path, "w") as f:
	for line in open(input_path):
		line = line.strip()
		
		if len(line) == 0:
			f.write('\n')
			continue

		f.write(f"{line}[PAR]\n")

import argparse
import run_tld

parser = argparse.ArgumentParser()
parser.add_argument("-s",default="C:/Users/heqin/PycharmProjects/TLD-TCN/src/dataset/surfer",help="the path of images")
parser.add_argument("-p", default="C:/Users/heqin/PycharmProjects/TLD-TCN/parameters.yml", help="parameter file")

if __name__ == '__main__':
  args = parser.parse_args()
  run = run_tld.RunTld(args)
  run.startTLD()

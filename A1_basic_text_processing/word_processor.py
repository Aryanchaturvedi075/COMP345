import re

file_path = '20000_tweets.txt'

f = open(file_path, "r", encoding='utf-8')

line_count = 1
top_tweets = []
for tweet in f:
#   print("### Tweet", line_count, "#####")
#   print(tweet)

  top_tweets.append(tweet)
  line_count += 1
  if line_count > 10:
    break
f.close()

s, c, p, op  = r"[+-−]?", r"[$€]?", r"%?", r"[eE/^*_:]?"
gi = r"\d{1,3}(?:,\d{3})+"
grp = r"(?:"+ gi +r"|\d+)(?:\.\d+)?"
sgn = s + grp
cur = s + c + grp + c + p
opn = sgn + op + sgn
bcp = s + r"\(" + cur + p + r"\)"


patterns = [
    r"https?://(?:www\.)?\S+" ,
    r"\w+(?=n['’]t\b)" ,
    r"n['’]t\b" ,
    r"\w+(?=['’]\w{1,2})" ,
    r"['’]\w{1,2}" ,
    r"\w+-\w+" ,
    r"(?:" + bcp + r"[,;\-−])+" + bcp ,
    s + c + r"\(" + opn + r"\)" + c + p ,
    r"(?:\d+[_.:/\-−]){2,}\d+" , 
    s + c + grp + op + sgn + c + p ,
    bcp ,
    cur ,
    r"[@#]?\w+" ,
    r"[\"\(\)\*\+\.\?\[\\\]\^\{\|\}]+" ,
    r"[!#&'-/<=>@$€]+" ,
    r"[%:;`~_,ʻ’“ˮ]" ,
    r"\S+"
]

def tokenize(text):
    # combined_pattern = '|'.join(f'(?:{pattern})' for pattern in patterns)
    search_ptrn = '|'.join(patterns)
    words = re.findall(search_ptrn, text.strip())
    return words

if __name__ == "__main__":
    test = "Covid-19 Economic Response: Cancel Student Loans by Executive Order. - Sign the Petition! https://t.co/BnPXWHv5cr via @Change wouldn't you'll I'm"
    print(tokenize(test))
    test = r" 123 -42 1e6 1.23e-4 1_000_000 1,234,567 1,234.56 01/21/2024 13:45 13:45:45:45 (1)−(2.3)-(3);($4) 1-2-3 123.45 .45 50.5% 5+3 1/2 10^6 $123 €12,345,678 123$ 1.2.3.4.454.45 1_234 1:302"
    print(tokenize(test))
    test = " Covid-19 wouldn't couldn't can't don't won't you'll I'm I've I'll I'd" 
    print(tokenize(test))
    test = "@NBCSAthletics Ya just knew the season wouldn’t go by without some bench clearing brawls... Covid or not. Behaviors can’t be changed because of rules, sadly"
    print(tokenize(test))
    tokenized_top_tweets = [tokenize(tweet) for tweet in top_tweets]
    for tokenized_tweet in tokenized_top_tweets:
        print(tokenized_tweet)
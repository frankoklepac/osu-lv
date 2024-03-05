def averageWordCount(sms_list):
    sum = 0
    for sms in sms_list:
        sum += len(sms.split())
    return (sum/len(sms_list)).__round__(2)

def spamEnding(sms_list):
    spam_ending = 0
    for sms in sms_list:
        if sms.endswith("!\n"):
            spam_ending += 1
    return spam_ending

fhead = open("LV1\SMSSpamCollection.txt", "r")
ham_sms = []
spam_sms = []
for line in fhead:
    if line.startswith("ham"):
        ham_sms.append(line)
    else:
        spam_sms.append(line)

print("Average word count for ham messages: ", averageWordCount(ham_sms))
print("Average word count for spam messages: ", averageWordCount(spam_sms))
print("Number of spam messages ending with !: ", spamEnding(spam_sms))
from numpy import size
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from transformers import pipeline

texts = ["""
         Dante was born in Florence, Republic of Florence, in what is now Italy. The exact date of his birth is unknown, although it is generally believed to be around 1265.[16] This can be deduced from autobiographic allusions in the Divine Comedy. Its first section, the Inferno, begins, "Nel mezzo del cammin di nostra vita" ("Midway upon the journey of our life"), implying that Dante was around 35 years old, since the average lifespan according to the Bible (Psalm 89:10, Vulgate) is 70 years; and since his imaginary travel to the netherworld took place in 1300, he was most probably born around 1265. Some verses of the Paradiso section of the Divine Comedy also provide a possible clue that he was born under the sign of Gemini: "As I revolved with the eternal twins, I saw revealed, from hills to river outlets, the threshing-floor that makes us so ferocious" (XXII  151–154). In 1265, the sun was in Gemini between approximately 11 May and 11 June (Julian calendar).[17]

Dante claimed that his family descended from the ancient Romans (Inferno, XV, 76), but the earliest relative he could mention by name was Cacciaguida degli Elisei (Paradiso, XV, 135), born no earlier than about 1100. Dante's father, Alighiero di Bellincione,[18] was a White Guelph who suffered no reprisals after the Ghibellines won the Battle of Montaperti in the middle of the 13th century. This suggests that Alighiero or his family may have enjoyed some protective prestige and status, although some suggest that the politically inactive Alighiero was of such low standing that he was not considered worth exiling.[19]

Dante's family was loyal to the Guelphs, a political alliance that supported the Papacy and that was involved in complex opposition to the Ghibellines, who were backed by the Holy Roman Emperor. The poet's mother was Bella, probably a member of the Abati family.[18] She died when Dante was not yet ten years old. His father Alighiero soon married again, to Lapa di Chiarissimo Cialuffi. It is uncertain whether he really married her, since widowers were socially limited in such matters, but she definitely bore him two children, Dante's half-brother Francesco and half-sister Tana (Gaetana).[18]
""","""
Portrait of Dante, from a fresco in the Palazzo dei Giudici, Florence
Dante said he first met Beatrice Portinari, daughter of Folco Portinari, when he was nine (she was eight),[20] and he claimed to have fallen in love with her "at first sight", apparently without even talking with her.[21] When he was 12, however, he was promised in marriage to Gemma di Manetto Donati, daughter of Manetto Donati, member of the powerful Donati family.[18] Contracting marriages for children at such an early age was quite common and involved a formal ceremony, including contracts signed before a notary.[18] Dante claimed to have seen Beatrice again frequently after he turned 18, exchanging greetings with her in the streets of Florence, though he never knew her well.[22]
Years after his marriage to Gemma, he claims to have met Beatrice again; he wrote several sonnets to Beatrice but never mentioned Gemma in any of his poems. He refers to other Donati relations, notably Forese and Piccarda, in his Divine Comedy. The exact date of his marriage is not known; the only certain information is that, before his exile in 1301, he had fathered three children with Gemma (Pietro, Jacopo and Antonia).[18]

Dante fought with the Guelph cavalry at the Battle of Campaldino (11 June 1289).[23] This victory brought about a reformation of the Florentine constitution. To take part in public life, one had to enroll in one of the city's many commercial or artisan guilds, so Dante entered the Physicians' and Apothecaries' Guild. In the following years, his name is occasionally recorded as speaking or voting in the various councils of the republic. A substantial portion of minutes from such meetings in the years 1298–1300 was lost, however, so the true extent of Dante's participation in the city's councils is uncertain.
         """]


# Initialize the summarization pipeline
summarizer = pipeline('summarization')

text_len = len(texts[0])+len(texts[1])
summary_len = 0

# Summarize the text
summary = summarizer(texts, max_length=500, min_length=100, do_sample=False)
for s in summary:
    summary_len += len(s['summary_text'])
    print(s['summary_text'])
    print("")

print("text_len = {}", text_len)
print("summary_len = {}", summary_len)
# Print the summary


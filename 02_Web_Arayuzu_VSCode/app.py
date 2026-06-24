from flask import Flask, request, jsonify, render_template
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import re
from collections import Counter
import difflib

app = Flask(__name__)
app.json.ensure_ascii = False 

# =================================================================
# 1. YAPAY ZEKA MODELİ ENTEGRASYONU
# =================================================================
MODEL_YOLU = "onuru/tez-duygu-modelim"
print("Yapay Zeka Modeli ve Tercüman yükleniyor, lütfen bekleyin... ⏳")
tokenizer = AutoTokenizer.from_pretrained(MODEL_YOLU)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_YOLU)
print("Model başarıyla yüklendi! ✅")



# Türkçe ve İngilizce metin madenciliği için gereksiz kelimeler havuzu
STOPWORDS = set([
    # --- TÜRKÇE BAĞLAÇ, ZAMİR, EDAT VE ZARFLAR ---
    "bir", "ve", "ile", "için", "çok", "bu", "şu", "o", "da", "de", "ama", "fakat", "ki", "gibi",
    "daha", "en", "mi", "mı", "mu", "mü", "ise", "var", "yok", "olan", "oldu", "olarak", "diye",
    "kadar", "şey", "her", "tüm", "sonra", "önce", "kendi", "ben", "sen", "biz", "siz", "onlar",
    "bana", "sana", "bizi", "sizi", "onu", "bunu", "şunu", "ondan", "bundan", "şundan", "ya", "hem",
    "ne", "göre", "böyle", "şöyle", "öyle", "sadece", "yalnız", "ancak", "bile", "dahi", "artık",
    "hiç", "hep", "gerçekten", "oldukça", "baya", "epey", "neden", "niye", "nasıl", "hangi", "kim",
    "zaten", "belki", "sanki", "galiba", "keşke", "kesinlikle", "asla", "mutlaka", "neredeyse", 
    "tamamen", "hiçbir", "bazen", "adeta", "resmen",
    
    # --- TÜRKÇE ZAYIF İSİMLER VE GENEL NİTELEYİCİLER (Her konuda geçerler, hedef olamazlar) ---
    "zaman", "gün", "gece", "sabah", "akşam", "yıl", "ay", "hafta", "bugün", "yarın", "dün", "şimdi",
    "insan", "insanlar", "kişi", "kişiler", "adam", "kadın", "çocuk", "biri", "birileri",
    "olay", "durum", "konu", "husus", "mesele", "şekil", "biçim", "tür", "çeşit", "kere", "defa", "kez",
    "büyük", "küçük", "eski", "yeni", "zor", "kolay", "farklı", "aynı", "başka", "diğer", "bütün",
    "uzun", "kısa", "ağır", "hafif", "doğru", "yanlış", "gerçek", "sahte", "boş", "dolu",
    
    # --- TÜRKÇE FİİLLER (Toplu analizde "aldım", "ettim" çıkmasın diye) ---
    "yapmak", "etmek", "olmak", "gitti", "geldi", "dedi", "yaptı", "etti", "aldım", "verdim", 
    "yaptım", "ettim", "gittim", "geldim", "aldı", "verdi", "çıktı", "kaldı", "bak", "baktı",
    "yapıyor", "ediyor", "geliyor", "gidiyor", "oluyor", "alıyor", "veriyor", "istiyorum", "istemiyorum",
    "seviyorum", "sevmiyorum", "kullanıyorum", "kullandım", "düşünüyorum", "bekliyorum",

    # --- İNGİLİZCE BAĞLAÇ, ZAMİR, EDAT, ZARF VE FİİLLER ---
    "the", "and", "was", "for", "with", "this", "that", "but", "very", "too", "it", "is", "in", 
    "on", "of", "to", "are", "not", "have", "has", "had", "they", "you", "i", "we", "he", "she", 
    "my", "your", "his", "her", "our", "their", "me", "him", "them", "us", "be", "am", "do", "does", 
    "did", "a", "an", "at", "by", "from", "up", "down", "out", "about", "into", "over", "after", 
    "so", "then", "there", "here", "when", "where", "why", "how", "all", "any", "both", "each",
    "just", "now", "get", "got", "make", "made", "can", "will", "would", "could", "should", "as",
    "done", "doing", "went", "gone", "going", "came", "coming", "take", "took", "taking", "give", "gave",
    
    # --- İNGİLİZCE ZAYIF İSİMLER VE GENEL NİTELEYİCİLER ---
    "time", "day", "night", "morning", "evening", "year", "month", "week", "today", "tomorrow", "yesterday",
    "man", "woman", "person", "people", "guy", "guys", "child", "children", "someone", "anyone", "everyone",
    "thing", "things", "stuff", "matter", "case", "point", "way", "type", "kind", "times",
    "big", "small", "old", "new", "hard", "easy", "different", "same", "other", "another", "whole",
    "long", "short", "heavy", "light", "true", "false", "real", "fake", "empty", "full",
    
    # --- MEGA DUYGU, DURUM VE SIFAT FİLTRESİ (Hedef "Aspect" kelimeleri perdelemesinler diye) ---
    # Türkçe Negatif Sıfatlar:
    "kötü", "berbat", "çirkin", "sorun", "sıkıntı", "rezalet", "fena", "bayağı", "kötüydü", "sorunu",
    "rezil", "kepaze", "vasat", "amatör", "dandik", "özensiz", "çile", "eziyet", "hüsran", "yetersiz", 
    "saçma", "lüzumsuz", "gereksiz", "faydasız", "bozuk", "yavaş", "pis", "kaba", "pahalı", "kazık", 
    "defolu", "kırık", "hasarlı", "bayat", "soğuk", "gecikme", "ilgisiz", "leş", "çöp", "hata", "kusur", 
    "yalan", "sahte", "üzgün", "kırgın", "sinir", "öfke", "fiyasko", "illet", "lanet", "eksik", "noksan", 
    "kalitesiz", "dayanıksız", "hurda", "enkaz", "paslı", "küflü", "şikayet", "pişman", "mutsuz", "sıkıcı",
    "iğrenç", "korkunç", "saygısız", "kullanışsız",
    
    # Türkçe Pozitif Sıfatlar:
    "iyi", "harika", "güzel", "mükemmel", "muhteşem", "şahane", "süper", "iyiydi", "güzeli", "başarılı",
    "kaliteli", "hızlı", "temiz", "taze", "sağlam", "kibar", "ucuz", "uygun", "memnun", "lezzet", "tatlı", 
    "kullanışlı", "pratik", "enfes", "şirin", "tavsiye", "efsane", "efsanevi", "şık", "zarif", "nefis", 
    "inanılmaz", "harikulade", "faydalı", "yararlı", "etkili", "güvenilir", "dürüst", "samimi", "nazik", 
    "sıcak", "sevimli", "rahat", "konforlu", "kusursuz", "muazzam", "teşekkür", "tebrik",
    
    # İngilizce Negatif Sıfatlar:
    "bad", "terrible", "awful", "worst", "problem", "issue", "poor", "unhappy", "boring", "disgusting", 
    "horrible", "worthless", "disappointed", "unacceptable", "broken", "damaged", "crash", "lag", "slow", 
    "dirty", "ugly", "rude", "expensive", "garbage", "trash", "late", "cold", "useless", "scam", "fake", 
    "sad", "angry", "tired", "hate", "dislike", "complain", "fault", "mistake", "error", "wrong",

    # İngilizce Pozitif Sıfatlar:
    "good", "great", "awesome", "amazing", "best", "nice", "excellent", "perfect", "fantastic", "flawless", 
    "beautiful", "brilliant", "love", "like", "recommend", "satisfied", "happy", "fast", "clean", "fresh", 
    "cheap", "polite", "delicious", "helpful", "useful", "cute", "sweet", "joy", "excited", "brilliant"
])



def dinamik_neden_bul(metin_listesi):
    # Eğer liste boşsa hata vermeden direkt dön
    if not metin_listesi: 
        return "Belirgin bir faktör bulunamadı."
        
    kelime_listesi = []
    # Liste içindeki DÜZ METİNLERİ (string) tek tek okuyoruz
    for m in metin_listesi:
        # Metni küçük harfe çevir ve noktalama işaretlerini sil
        temiz = re.sub(r'[^\w\s]', '', str(m).lower())
        # Anlamlı kelimeleri (Stopwords hariç) listeye ekle (SET İLE TEKİLLEŞTİRİLDİ)
        kelime_listesi.extend(list(set([k for k in temiz.split() if k not in STOPWORDS and len(k) > 2])))
        
    # Eğer temizlikten sonra geriye kelime kalmadıysa
    if not kelime_listesi: 
        return "Belirgin bir faktör bulunamadı."

    # Kelimeleri say
    frekans = Counter(kelime_listesi)
    en_cok_gecen = frekans.most_common(1)[0][0]
    
    # Harf hatalarını ve ekleri toparla (kargo, kargom, kargolar vb.)
    toplam_frekans = 0
    for kelime, sayi in frekans.items():
        if kelime.startswith(en_cok_gecen[:4]) or difflib.SequenceMatcher(None, en_cok_gecen, kelime).ratio() > 0.75:
            toplam_frekans += sayi
            
    # Yüzdeyi hesapla
    yuzde = (toplam_frekans / len(metin_listesi)) * 100
    
    return f"Yorumların %{yuzde:.0f}'sinde ana faktör olarak '{en_cok_gecen.upper()}' öne çıkıyor."


# =================================================================
# 3. WEB SAYFASI VE API ROTALARI
# =================================================================
@app.route('/', methods=['GET'])
def ana_sayfa():
    return render_template('index.html')
# =================================================================
# YENİ NESİL TAHMİN MOTORU (TR/EN Çift Dil, Çoklu Neden, Akıllı Parçalayıcı)
# =================================================================
@app.route('/tahmin', methods=['POST'])
def tahmin_yap():
    try:
        veri = request.get_json()
        metin = veri.get('metin', '')
        if not metin:
            return jsonify({'hata': 'Metin boş.'}), 400
        
        # YENİ: Ön İşleme (Aşırı -> Çok çevrimi)
        temiz_metin = metin.replace("aşırı derecede", "çok").replace("Aşırı derecede", "Çok")
        temiz_metin = temiz_metin.replace("aşırı", "çok").replace("Aşırı", "Çok")

        # =================================================================
          # TOTAL (HOLİSTİK) METİN ANALİZİ
        # =================================================================
        

        
        
        

        # 1. AKILLI PARÇALAYICI (TR/EN Bağlaçlar)
        ayiricilar_regex = r'(\b(?:ne var ki|buna rağmen|bununla birlikte|sonuç olarak|bundan dolayı|bu yüzden|bu sebeple|hem de|ya da|on the other hand|as a result|for this reason|ama|fakat|lakin|ancak|yalnız|halbuki|oysa|oysaki|yine de|veya|yahut|veyahut|ve|ayrıca|üstelik|hatta|çünkü|zira|dolayısıyla|aksine|tersine|gelgelelim|and|but|or|nor|however|although|though|yet|nevertheless|nonetheless|whereas|while|except|otherwise|conversely|alternatively|plus|moreover|furthermore|additionally|because|since|therefore|thus|hence|consequently)\b|[,\.;!\?\|\n\t—–])'
        
        ham_parcalar = re.split(ayiricilar_regex, temiz_metin)
        parcalar = []
        gecici_cumle = ""
        bekleyen_ayirici = ""
        
        for p in ham_parcalar:
            if p is None: continue
            p_temiz = p.strip()
            if not p_temiz: continue
            
            if re.fullmatch(ayiricilar_regex, p_temiz):
                bekleyen_ayirici = p_temiz
            else:
                kelimeler = p_temiz.split()
                if not gecici_cumle:
                    gecici_cumle = (bekleyen_ayirici + " " + p_temiz).strip()
                    bekleyen_ayirici = ""
                elif len(kelimeler) <= 1:
                    if re.match(r'[,\.;!\?\|\n\t—–]', bekleyen_ayirici):
                        gecici_cumle += bekleyen_ayirici + " " + p_temiz
                    else:
                        gecici_cumle += " " + bekleyen_ayirici + " " + p_temiz if bekleyen_ayirici else " " + p_temiz
                    bekleyen_ayirici = ""
                else:
                    parcalar.append(gecici_cumle.strip())
                    gecici_cumle = (bekleyen_ayirici + " " + p_temiz).strip()
                    bekleyen_ayirici = ""
                    
        if gecici_cumle:
            son_cumle = gecici_cumle + bekleyen_ayirici
            parcalar.append(son_cumle.strip())
            
        parcalar = [p for p in parcalar if len(p.replace(',', '').replace('.', '').strip()) > 2]

        detayli_sonuclar = [] 

     # =================================================================
        # 2. MEGA SÖZLÜKLER (Pozitif ve Negatif Olarak İkiye Ayrıldı!)
        # =================================================================
        negatif_ifadeler =[
            # TR/EN Negatif
            "mutsuz", "sıkıcı", "bunal", "daral", "iğren", "berbat", "rezalet", "korkunç", "paramparça", "saygısız", "kullanışsız", 
            "sevmi", "sevme", "seveme", "nefret", "şikayet", "pişman", "boz", "çalışmı", "çalışma", "kasıy", "donuy","isteme","istemiyo",
            "çök", "yavaş", "kötü", "pis", "çirkin", "kaba", "pahalı", "kazık", "defolu", "kırık", "hasarlı", "bayat","gitme","gelme"
            "soğuk", "gecik", "ilgisiz", "leş", "çöp", "sorun", "sıkıntı", "hata", "kusur", "yalan", "sahte", "dolandır", 
            "rezil", "kepaze", "vasat", "amatör", "dandik", "özensiz", "çile", "eziyet", "hüsran", "yetersiz", "saçma", 
            "lüzumsuz", "gereksiz", "faydasız", "karanlık", "bulanık", "üzgün", "kırgın", "sinir", "öfke", "bık", "yorul", 
            "keder", "ağla", "acı", "fiyasko", "illet", "lanet", "kavga", "tartış", "döv", "vur", "çal", "kaybet", "bozul", 
            "yanıl", "şaşır", "kandır", "aldat", "ayrıl", "terk","unhappy", "boring", "disgusting", "terrible", "awful", "horrible", "worthless", "disappointed", "unacceptable", 
            "hate", "dislike", "complain", "broken", "damaged", "crash", "lag", "slow", "bad", "dirty", "ugly", "rude", 
            "expensive", "garbage", "trash", "late", "cold", "useless", "scam", "fake", "issue", "sad", "angry", "tired", "cry"
            "üzül", "yas ", "matem", "hayal kırık", "vicdan azab", "kahrol", "yıkıl", "mahvol", "tüken", "pes et", "boğul", 
            "sıkıl", "stres", "kaygı", "endişe", "panik", "kork", "ürk", "titre", "dehşet", "kabus", "travma", "depres", 
            "anksiyet", "şüphe", "vesvese", "paranoy", "yalnız", "çaresiz", "umutsuz", "karamsar", "bedbin", "zavallı", 
            "ezik", "dışlan", "aşağılan", "kızgın", "kızd", "kin", "intikam", "haset", "kıskan", "düşman", "çıldır", "delir", 
            "kudur", "cinnet", "bağır", "çağır", "çatış", "saldır", "kır", "parçala", "dağıt", "mahvet", "yok et", "öldür", 
            "katlet", "cinayet", "zulüm", "zalim", "işkence", "bela", "musibet", "zıkkım", "küfür", "küfr", "hakaret", 
            "beddua", "suçla", "yargıla", "kına", "isyan", "itiraz", "tiksin", "facia", "felaket", "mantıksız", "anlamsız", 
            "boş", "aptal", "salak", "ahmak", "cahil", "yobaz", "bencil", "egoist", "kibirli", "küstah", "şımarık", "nankör", 
            "hain", "ikiyüzlü", "iftira", "riya", "ahlaksız", "terbiyesiz", "edepsiz", "vicdansız", "insafsız", "gaddar", 
            "acımasız", "kirli", "pasaklı", "kokuş", "zayıf", "güçsüz", "başarısız", "yenil", "yanlış", "yırtık", "sökük", 
            "ezik", "çizik", "yamuk", "patlak", "çatlak", "leke", "eksik", "noksan", "kalitesiz", "sağlam deği", "dayanıksız", 
            "kullanılamaz", "hurda", "enkaz", "paslı", "küflü", "koptu", "kopuy", "çekmiy", "girmiy", "açılmı", "açılma", 
            "bağlanamı", "ısın", "kapanı", "kapat", "virüs", "bug", "silin", "sıfırland", "gelmedi", "getirmedi", "ulaşmadı", 
            "kaybol", "farklı geldi", "iptal", "reddedil", "açmıy", "ukala", "umursamaz", "ciddiyetsiz", "mağdur", "çiğ", 
            "yanık", "kokuy", "tuzlu", "şekersiz", "kıl", "tüy", "böcek", "sinek", "zehirl", "midem", "kus", "hijyensiz", 
            "soygun", "hırsız", "soydu", "fahiş", "zarar", "battı", "tahammül edemi", "katlanamı", "uzak dur", "kaçın", 
            "asla", "sakın", "depress", "sorrow", "grief", "mourn", "weep", "sob", "tear", "pain", "hurt", "ache", "suffer", 
            "agon", "miser", "tragic", "disappoint", "regret", "guilt", "shame", "embarrass", "humiliat", "lonel", "alone", 
            "isolat", "hopeless", "despair", "pessimis", "helpless", "scare", "fright", "terr", "panic", "dread", "anxi", 
            "worry", "nervous", "paranoi", "exhaust", "tire", "fatigue", "drain", "burnout", "give up", "quit", "fail", 
            "lose", "lost", "angr", "anger", "mad", "furi", "rage", "wrath", "despise", "detest", "loath", "abhor", "jealous", 
            "envy", "revenge", "enem", "hostil", "aggress", "violenc", "violent", "attack", "fight", "argu", "conflict", 
            "hit", "beat", "strike", "smash", "break", "destroy", "ruin", "crush", "kill", "murder", "dead", "death", "die", 
            "fatal", "lethal", "toxic", "poison", "cruel", "brutal", "savag", "tortur", "abus", "harass", "bully", "threat", 
            "curse", "swear", "insult", "offend", "blame", "horrend", "dread", "appall", "atroci", "worst", "gross", "nasty", 
            "foul", "vile", "hideous", "filth", "mess", "stink", "pointless", "meaningless", "dump", "stupid", "dumb", "idiot", 
            "fool", "moron", "ignor", "arrogant", "selfish", "greed", "stubborn", "deceiv", "cheat", "betray", "traitor", 
            "hypocrit", "evil", "wicked", "impolit", "disrespect", "weak", "feeble", "poor", "flaw", "fault", "mistake", 
            "error", "wrong", "incorrect", "invalid", "ill", "sick", "diseas", "symptom", "torn", "dent", "crack", "stain", 
            "miss", "lack", "fragile", "rust", "mold", "disconnect", "drop", "overheat", "freeze", "froze", "differ", 
            "cancel", "reject", "ignor", "unprofession", "victim", "mock", "liar", "lie", "raw", "burnt", "odor", "hair", 
            "fly", "puke", "unhygien", "rob", "thief", "steal", "stole", "overcharg", "ripoff", "fraud", "bankrupt", 
            "devastat", "frustrat", "crazy", "insane", "unbeliev", "unbear", "avoid", "never"
        ]

        pozitif_ifadeler =[
            # TR/EN Pozitif
            "mükemmel", "muazzam", "harika", "şahane", "efsane", "kusursuz", "bayıl", "seviy", "sevdi", "beğen", "başarılı","ister",
            "kaliteli", "güzel", "süper", "iyi", "hızlı", "temiz", "taze", "sağlam", "kibar", "ucuz", "uygun", "memnun","istiyor",
            "lezzet", "tatlı", "kullanışlı", "pratik", "enfes", "şirin", "tavsiye", "efsanevi", "şık", "zarif", "nefis", "sürükle"
            "muhteşem", "inanılmaz", "harikulade", "faydalı", "yararlı", "etkili", "güvenilir", "dürüst", "samimi", "nazik", 
            "sıcak", "sevimli", "rahat", "konforlu", "ferah", "aydınlık", "orijinal", "doyurucu", "kral", "coşku", "çoşku", 
            "heyecan", "mutlu", "sevin", "neşeli", "huzur", "aşık", "keyif", "öv", "takdir", "minnet", "şans", "harbi", "şampiyon",
            "anlaş", "uzlaş", "barış", "çöz", "kurtar", "kazan", "geliş", "iyileş", "destek", "yardım", "başar", "kutla", "sarıl",
            "excellent", "amazing", "awesome", "perfect", "fantastic", "flawless", "beautiful", "brilliant", "love",
            "like", "recommend", "satisfied", "happy", "good", "great", "fast", "clean", "fresh", "cheap", "polite", 
            "delicious", "helpful", "useful", "cute", "sweet", "nice", "joy", "excited""sev", "aşk", "tutku", "sadık", "sadakat", "güven", "vefa", "şefkat", "merhamet", "cesur", "kahraman", "zeki", 
            "deha", "dahi", "yetenek", "mucize", "hayran", "mest", "coş", "ferahla", "rahatla", "dingin", "pozitif", 
            "sevecen", "gülüm", "kahkaha", "tebessüm", "neşel", "şevk", "heves", "gurur", "özen", "ilgi", "alak", "on numara", 
            "10 numara", "olağanüstü", "pürüzsüz", "eşsiz", "benzersiz", "rakipsiz", "estetik", "yakışıkl", "tatlış", "minnoş", 
            "hoş", "leziz", "gurme", "hesaplı", "ekonomik", "bedava", "hediye", "ikram", "kıyak", "jest", "lütuf", "müthiş", 
            "paha biçilemez", "değerli", "kıymetli", "altın", "elmas", "dayanıklı", "uzun ömürlü", "pırıl", "ışıl", "parlak", 
            "canlı", "taptaze", "yepyeni", "bambaşka", "teşekkür", "tebrik", "alkış", "övgü", "onayla", "kabul et", "koru", 
            "kucakla", "öp", "şifa", "bereket", "bolluk", "cennet", "melek", "nimet", "lüks", "konfor", "efendi", "saygıl", 
            "anlayış", "hoşgörü", "sempat", "çekici", "büyüleyic", "göz alıcı", "şıkır", "tertemiz", "mis", "cillop", "onay", 
            "tatminkar", "beklentimi karşı", "ötesi", "aşmış", "yakıyo", "bayıld", "hoşuma", "iyi ki", "helal", "bravo", "tebrik",
            "proud", "brave", "smart", "clever", "genius", "talent", "miracl", "admir", "enchant", "calm", "peace", "positiv", 
            "smile", "laugh", "cheer", "eager", "enthusias", "thrill", "glad", "delight", "care", "honest", "kind", "legend", 
            "extraordinar", "smooth", "uniqu", "matchless", "unrival", "stylish", "aesthetic", "handso", "gorgeou", "stunn", 
            "adorabl", "yum", "tast", "gourmet", "econom", "free", "gift", "bonus", "incredibl", "valuabl", "preciou", "gold", 
            "diamond", "quality", "solid", "durabl", "long-last", "shin", "bright", "vivid", "thank", "congrat", "applaud", 
            "prais", "approv", "accept", "support", "protect", "sav", "hug", "embrac", "kiss", "passion", "loyal", "trust", 
            "reliabl", "safe", "fabulou", "marvel", "splendid", "superb", "spectacular", "phenomenal", "outstand", "glorious", 
            "magical", "heaven", "angel", "bless", "luxur", "respect", "toleran", "sympath", "attract", "fascinat", "charm", 
            "impress", "satisfy", "beyond expect", "well done", "kudo"
        ]
        
        # Eski yapıyı bozmamak için iki listeyi birleştiriyoruz
        duygu_ifadeleri = negatif_ifadeler + pozitif_ifadeler 

        kesin_copluk = [
            # TR/EN Çöplük (Bağlaçlar, Zarflar, Kesinlikle Hedef Olamayacaklar)
            'bugün', 'yarın', 'şimdi', 'dün', 'şu an', 'o an', 'şuan', 'önce', 'sonra', 'yine', 'de', 'iyiyim', 'kötüyüm','yani',
            'çok', 'bir', 'hiç', 'en', 'daha', 'baya', 'aşırı', 'az', 'fazla', 'oldukça', 'gayet', 'sadece', 'yalnızca', 'biraz', 'yine','yinede',
            'tüm', 'bütün', 'bazı', 'her', 'hep', 'kesinlikle', 'mutlaka', 'gerçekten', 'neredeyse', 'tamamen', 'epey', 'hayli', 
            'belki', 'sanki', 'galiba', 'sanırım', 'umarım', 'keşke', 'adeta', 'resmen', 'asla', 'bazen',
            'için', 'gibi', 'diye', 'olan', 'olarak', 'ise', 'ile', 'var', 'yok', 'artık', 'hala', 'henüz', 'hemen', 'derhal', 
            'göre', 'doğru', 'karşı', 'ne', 'nasıl', 'niçin', 'kim', 'nerede', 'nereye', 'nereden', 'hangi', 'kaç', 'falan', 'filan',
            've', 'ama', 'fakat', 'lakin', 'ancak', 'yalnız', 'oysa', 'oysaki', 'halbuki', 'çünkü', 'zira', 'veya', 'yahut', 'veyahut', 'ya', 'da', 'hem',
            'the', 'a', 'an', 'my', 'is', 'are', 'am', 'it', 'this', 'that', 'very', 'much', 'too', 'so', 'really', 'just',
            'all', 'any', 'some', 'more', 'less', 'was', 'were', 'be', 'been', 'has', 'have', 'had', 'do', 'does', 'did',
            'to', 'of', 'in', 'on', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during',
            'and', 'but', 'or', 'nor', 'because', 'since', 'although', 'though', 'however', 'therefore', 'thus'
            'olduğu', 'olmak', 'mış', 'miş', 'muş', 'müş', 'mu', 'mü', 'mı', 'mi', 'idi', 'imiş', 'ise', 'dir', 'dır', 'dur', 'dür',
            'the', 'a', 'an', 'my', 'is', 'are', 'am', 'it', 'this', 'that', 'very', 'much', 'too', 'so', 'really', 'just',
            'all', 'any', 'some', 'more', 'less', 'was', 'were', 'be', 'been', 'has', 'have', 'had', 'do', 'does', 'did',
            'to', 'of', 'in', 'on', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during'
        ]

        zayif_isimler = [
            # TR/EN Zayıf İsimler (Zaman, Soyut Kavramlar, Hitaplar)
            'ben', 'sen', 'o', 'biz', 'siz', 'onlar', 'benim', 'senin', 'onun', 'bizim', 'sizin', 'onların', 'kendi', 'kendisi',
            'durum', 'olay', 'konu', 'husus', 'mesele', 'şey', 'şeyler', 'kısım', 'taraf', 'açı', 'yön', 'sebep', 'neden', 'sonuç', 
            'amaç', 'gaye', 'biçim', 'şekil', 'hal', 'tür', 'çeşit', 'kere', 'defa', 'sefer', 'kez', 'adet', 'tane', 'boy', 'kat',
            'abi', 'abla', 'bey', 'hanım', 'efendi', 'kardeş', 'dostum', 'kanka', 'aga', 'usta', 'hacı', 
            'arkadaş', 'insan', 'insanlar', 'kişi', 'kişiler', 'biri', 'birileri',
            'bugün', 'yarın', 'dün', 'şimdi', 'şu', 'an', 'sonra', 'önce', 'zaman', 'gün', 'gece', 'sabah', 'akşam', 'öğle',
            'saniye', 'dakika', 'saat', 'hafta', 'ay', 'yıl', 'asır', 'devir', 'mevsim', 'günlerde', 'aylarda', 'yıllarda',
            'today', 'tomorrow', 'yesterday', 'now', 'then', 'here', 'there', 'when', 'where', 'why', 'how', 'who', 'what',
            'thing', 'things', 'stuff', 'matter', 'case', 'point', 'guy', 'guys', 'man', 'person', 'people', 'time', 'day'
        ]
        # =================================================================
        # MEGA KATEGORİ HARİTALAMA SÖZLÜĞÜ (TR/EN - Omni-Domain Ontology)
        # =================================================================
        kategori_sozlugu = {
            "Psikoloji / Kişisel Durum (Psychology / State)": ["ben", "psikoloji", "ruh", "his", "duygu", "can", "kalp", "akıl", 
             "zihin", "hayat", "yaşam", "gün", "umut", "sinir", "stres", "moral", "keyif", "me",
             "mind", "soul", "feeling", "emotion", "heart", "life", "day", "hope", "stress", "mood", "myself",
             "mutlu", "üzgün", "üzüntü", "kork", "heyecan", "öfke", "kaygı", "endişe", "panik", "travma", 
             "depres", "anksiyet", "ağla", "gülüm", "kahkaha", "yorgun", "bitkin", "huzur", "yalnız", 
             "özgüven", "cesaret", "şüphe", "takınt", "heves", "sabır", "sabr", "tahammül", "vicdan", 
             "pişman", "şok", "nefret", "sevg", "aşk", "tutku", "bunald", "darald", "rahatla", "tedirgin",
             "telaş", "gurur", "kıskan", "özlem", "hasret", "neşe", "coşku", "keder", "acı", "bık", "usand",
             "happ", "sad", "fear", "scar", "excit", "angr", "anger", "anxi", "panic", "trauma"],




            
            "Spor / Sağlık (Sports / Health)": ["spor", "maç", "futbol", "basketbol", "takım", "oyuncu", "antrenman", "hakem", 
            "kupa", "lig", "şampiyon", "doktor", "hastane", "ilaç", "tedavi", "ağrı", "hastalık", "sağlık", "sport", "match", 
            "football", "soccer", "basketball", "team", "player", "referee", "cup", "league", "champion", "doctor", "hospital", 
            "medicine", "pain", "sick", "illness", "health", "fitness",
            "voleybol", "teni", "yüzme", "yüzüc", "koşu", "atlet", "jimnast", "güreş", "boks", "turnuv", 
            "madaly", "antrenör", "stadyum", "saha", "file", "raket", "gol", "skor", "forma", "krampon",  
            "taraftar", "seyirc", "ofsayt", "penalt", "transfer", "idman", "hasta", "tahlil", "muayen", 
            "ameliyat", "cerrah", "hemşir", "ebe", "klinik", "reçet", "eczan", "vitamin", "sızı", "kan", 
            "nabız", "tansiyon", "kriz", "ateş", "öksür", "hapşır", "diyet", "beslen", "kilo", "zayıf", 
            "obez", "virüs", "mikrop", "enfeksiy", "aşı", "serum", "röntgen", "ultrason", "terap", "taburcu",
            "sendrom", "kalori", "protein","volley", "tennis", "swim", "run", "athlet", "gym", "wrestl", "box", "tourna", "medal", 
            "coach", "stadium", "field", "court", "racket", "goal", "score", "jersey", "cleat", "fan", 
            "spectat", "offside", "penalt", "transfer", "train", "patient", "exam", "surg", "nurs", 
            "clinic", "prescrip", "pharmac", "pill", "vitamin", "ache", "blood", "pulse", "tension", 
            "fever", "cough", "sneez", "diet", "nutrit", "weight", "obes", "virus", "germ", "infect", 
            "vaccin", "serum", "x-ray", "ultrasound", "therap", "syndrome", "calori"], 
 


            
            "Siyaset / Toplum (Politics / Society)": ["siyaset", "parti", "seçim", "oy", "bakan", "başkan", "hükümet", 
            "devlet", "kanun", "yasa", "haber", "toplum", "halk", "ülke", "vatan", "adalet", "gündem", "politics", "party", 
            "election", "vote", "minister", "president", "government", "state", "law", "news", "society", "people", "country", 
            "nation", "justice","meclis", "milletvekil", "vekil", "muhalefet", "iktidar", "rejim", "demokras", "cumhuriyet", 
            "diktatör", "diplomat", "büyükelç", "elçi", "konsolos", "asker", "polis", "jandarm", 
            "mahkem", "hakim", "savcı", "avukat", "dava", "ceza", "suç", "hapishan", "cezaev", 
            "yoksulluk", "kriz", "grev", "protest", "eylem", "yürüyüş", "vatandaş", "yurttaş", 
            "mülteci", "sığınmac", "göçmen", "ırkç", "faşist", "komünist", "sosyalist", "sağcı", "solcu", 
            "muhafazakar", "laik", "şeriat", "darbe", "terör", "savaş", "barış", "antlaşma", "sözleşme", 
            "anayasa", "tasarı", "kararname", "genelge", "sendika", "dernek", "vakıf", "miting", "sandık",
            "yolsuzluk", "rüşvet", "medya", "gazete", "sansür", "özgürlük", "eşitlik",
            "parliament", "congress", "senat", "deput", "represent", "opposit", "power", "regime", 
            "democrac", "republic", "dictat", "diplom", "ambassad", "consul", "militar", "arm", 
            "polic", "cop", "court", "judg", "prosecut", "lawy", "attorney", "case", "punish", 
            "crime", "prison", "jail", "povert", "cris", "strike", "protest", "action", "march", 
            "citizen", "refug", "immigr", "migrant", "racist", "fascist", "communis", "socialis", 
            "rightist", "leftist", "conservat", "secular", "sharia", "coup", "terror", "war", "peace", 
            "treat", "pact", "constitut", "bill", "act", "decre", "union", "associat", "foundat", 
            "corrupt", "bribe", "media", "newspaper", "censor", "freedom", "equal"], 






            
            "Sanat / Eğlence (Art / Entertainment)": ["sanat", "film", "dizi", "sinema", "müzik", "şarkı", "kitap", "roman", 
            "yazar", "oyuncu", "aktör", "tiyatro", "konser", "sergi", "resim", "art", "movie", "film", "series", "cinema", "music", 
            "song", "book", "novel", "author", "actor", "actress", "theater", "concert", "exhibition", "painting","yönetmen", "senary",
            "senaris", "kurgu", "sahne", "sezon", "bölüm", "belgesel","animasyon", "vizyon", "gişe", "festival", "albüm", "klip", "melod", "ritim", "nota", 
            "enstrüman", "gitar", "piyano", "davul", "vokal", "koro", "dinle", "şiir", "şair", 
            "hikay", "öykü", "masal", "edebiyat", "heykel", "ressam", "tuval", "fırça", "galer", 
            "eğlenc", "gösteri", "komed", "dram", "kurgu", "bilim kurgu", "korku", "gerilim",
            "fragman", "kamera arkası", "prodüksiyon", "ödül", "oscar", "prömiyer", "biyografi",
            "direct", "script", "scene", "season", "episod", "documentar", "animat", "box offic", 
            "fest", "album", "clip", "melod", "rhythm", "note", "instrument", "guitar", "piano", 
            "drum", "vocal", "choir", "listen", "poet", "poem", "stor", "tale", "literat"],






            
            "Doğa / Hayvanlar (Nature / Animals)": ["hayvan", "köpek", "kedi", "kuş", "balık", "doğa", "ağaç", "orman", "deniz",
            "hava", "su", "çevre", "manzara", "animal", "dog", "cat", "bird", "fish", "nature", "tree", "forest", "sea", "weather",
            "water", "environment", "pet","at", "inek", "aslan", "kaplan", "böcek", "yılan", "fare", "evcil", "veteriner", "mama", 
            "tasma", "pati", "kuyruk", "kanat", "tüy", "dağ", "tepe", "nehir", "ırmak", "göl", "okyanus", 
            "çöl", "vadi", "mağar", "toprak", "taş", "kaya", "yaprak", "çiçek", "bitk", "çimen", "ot", 
            "iklim", "yağmur", "kar", "fırtın", "rüzgar", "güneş", "bulut", "sıcak", "soğuk", "don", 
            "deprem", "afet", "sel", "tsunami", "ekoloj", "kirlilik", "orman", "fidan", "tohum", "doğal",
            "horse", "cow", "lion", "tiger", "insect", "bug", "snake", "mous", "vet", "food", 
            "leash", "paw", "tail", "wing", "feather", "mountain", "hill", "river", "lake", "ocean", 
            "desert", "valley", "cave", "soil", "dirt", "rock", "stone", "leaf", "flower", "plant", 
            "grass", "weed", "climat", "rain", "snow", "storm", "wind", "sun", "cloud", "hot", "cold"],




            
            "Eğitim / Okul (Education)": ["okul", "üniversite", "lise", "öğretmen", "hoca", "öğrenci", "ders", "sınav", "not", 
            "eğitim", "tez", "jüri", "sınıf", "kampüs", "school", "university", "college", "teacher", "professor", "student", 
            "lesson", "exam", "grade", "education", "thesis", "jury", "class", "campus","akadem", "fakült", "enstitü", "bölüm", "dekan", "rektör", "asistan", "danışman", 
            "müfredat", "ödev", "proje", "sunum", "laboratuvar", "deney", "araştırm", "makale","okul",
            "bildiri", "mezun", "diplom", "karne", "burs", "kredi", "harç", "kayıt", "staj", 
            "devamsız", "vize", "final", "bütünleme", "mülakat", "ilkokul", "ortaokul", "kreş", 
            "anaokul", "yükseklisan", "doktora", "tahta", "kantin", "yurt", "kütüphan", "seminer"
            "academ", "facult", "institut", "departmen", "dean", "rector", "assist", "advisor", 
            "curriculum", "homework", "assign", "project", "present", "lab", "experimen", "research", 
            "article", "paper", "graduat", "diplom", "report card", "scholar", "tuition", "enroll", 
            "intern", "absent", "midterm", "final", "makeup", "interview", "kindergarten", "primary", 
            "secondar", "master", "phd", "board", "canteen", "dorm", "librar", "seminar"],




            
            "Günlük Eşyalar / Objeler (Everyday Items)": ["sigara", "yastık", "yatak", "kıyafet", "elbise", "ayakkabı", "çanta", 
            "saat", "gözlük", "kalem", "defter", "masa", "koltuk", "eşya", "obje", "cigarette", "pillow", "bed", "clothes", "dress",
            "shoes", "bag", "watch", "glasses", "pen", "notebook", "table", "sofa", "item", "object", "thing","sandalye", "dolap", "sehpa", 
            "halı", "perde", "ayna", "silgi", "kağıt", "makas", "bant", 
            "dosya", "cüzdan", "şemsiy", "anahtar", "yüzük", "kolye", "küpe", "kemer", "şapka", 
            "eldiven", "atkı", "bere", "bardak", "tabak", "çatal", "kaşık", "bıçak", "tencer", "tava", 
            "şişe", "kutu", "sepet", "fırça", "süpürg", "havlu", "sabun", "çakmak", "cımbız", "toka", 
            "bavul", "çadır", "matara", "termos", "kupa", "tepsi", "kase", "minder", "battaniy",
            "yorgan", "çarşaf", "aksesuar", "takı", "biblo", "vazo", "çerçeve",
            "chair", "desk", "closet", "cabinet", "carpet", "rug", "curtain", "mirror", "eras", 
            "paper", "scissor", "tape", "file", "wallet", "umbrella", "key", "ring", "necklac", 
            "earring", "belt", "hat", "cap", "glove", "scarf", "beanie", "cup", "mug", "glass", 
            "plate", "bowl", "fork", "spoon", "knife", "pot", "pan", "bottl", "box", "basket", 
            "brush", "broom", "towel", "soap", "lighter", "tweez", "suitcase", "tent", "flask", 
            "thermos", "tray", "cushion", "blanket", "quilt", "sheet", "accessor", "jewel", "vase", "frame"],




            
            "Donanım / Ürün (Hardware / Product)": ["ekran", "kamera", "batarya", "şarj", "kasa", "tuş", "hafıza", "işlemci", "cihaz", 
            "ürün", "malzeme", "kumaş", "beden", "renk", "kalıp", "telefon", "bilgisayar", "araba", "motor", "screen", "camera", "battery",
            "charge", "case", "button", "memory", "processor", "device", "product", "material", "fabric", "size", "color", "phone", "computer", 
            "car", "engine","anakart", "klavye", "hoparlör", "mikrofon", "kulaklık", "kablo", "adaptör", "soket", 
            "fiş", "priz", "sensör", "lens", "panel", "çip", "disk", "soğutuc", "monitör", "televizyon", 
            "alet", "makin", "parça", "donanım", "teker", "lastik", "fren", "direksiyon", "vites", 
            "kaport", "tampon", "far", "silecek", "doku", "yüzey", "kalit", "marka", "model", 
            "orijinal", "çakma", "replik", "ambalaj", "etiket", "kılıf", "koruyuc", "kordon",
            "tablet", "laptop", "modem", "yönlendiric", "klima", "komb", "beyaz eşya", "buzdolab",
            "motherboard", "keyboard", "speaker", "mic", "headphon", "earphon", "cable", "wire", 
            "adapt", "socket", "plug", "outlet", "sensor", "lens", "panel", "chip", "disk", "drive", 
            "fan", "cool", "monitor", "televis", "tool", "machin", "part", "wheel", "tire", "brake", 
            "steer", "gear", "hood", "bumper", "headlight", "wiper", "textur", "surfac", "qualit", 
            "brand", "model", "origin", "replic", "packag", "label", "cover", "protect", "strap",
            "tablet", "laptop", "modem", "router", "ac ", "air cond", "fridge"],




            
            "Yazılım / Dijital (Software / Digital)": ["uygulama", "sistem", "yazılım", "hız", "fps", "kasma", "donma", "çökme", 
            "güncelleme", "arayüz", "internet", "bağlantı", "ping", "oyun", "program", "app", "application", "system", "software", 
            "speed", "lag", "crash", "freeze", "update", "interface", "connection", "game", "website","sürüm", "versiy", "tarayıc", "sunuc", "bulut", "ağ", "wifi", "fiber", "kota", "şebek", 
            "kopma", "koptu", "hata", "virüs", "kasıy", "donuy", "çöküy", "gecikm", "kod", "algoritm", 
            "veri", "taban", "site", "web", "link", "profil", "hesap", "şifr", "parol", "giriş", 
            "tasarım", "menü", "buton", "sekm", "bildirim", "mesaj", "grafik", "çözünürlük", "piksel", 
            "hile", "bot", "seviye", "karakter", "oyuncu", "kurulum", "yükle", "indir", "format",
            "yapay zeka", "yedek", "sunucu", "modem", "güvenlik", "erişim", "premium", "reklam",
            "version", "brows", "serv", "cloud", "network", "data", "site", "link", "url", "error", 
            "bug", "glitch", "virus", "malwar", "delay", "drop", "disconnect", "code", "algorithm", 
            "databas", "profil", "account", "password", "login", "logout", "design", "menu", "button", 
            "tab", "notific", "messag", "graphic", "resolut", "pixel", "cheat", "bot", "level", 
            "character", "player", "multiplay", "install", "download", "load", "backup", "secur", 
            "access", "premium", "ad ", "ads"],
            





            "Lojistik / Hizmet (Logistics / Service)": ["kargo", "kurye", "teslimat", "paket", "kutu", "müşteri", "hizmet", "destek", 
            "iade", "değişim", "garanti", "servis", "çalışan", "personel", "temsilci", "satıcı", "müdür", "cargo", "courier", "delivery", 
            "package", "box", "customer", "service", "support", "return", "exchange", "warranty", "staff", "employee", "representative", 
            "seller", "manager","lojistik", "nakliy", "taşıma", "taşınd", "ulaştır", "şube", "dağıtım", "takip", "barkod", 
            "ambalaj", "poşet", "sevk", "sipariş", "stok", "depo", "tedarik", "fatur", "fiş", "makbuz", 
            "şikayet", "talep", "yardım", "çağrı", "iletişim", "irtibat", "muhatap", "yetkili", 
            "uzman", "teknisyen", "usta", "tamir", "bakım", "onarım", "kurulum", "montaj", "şoför",
            "gecik", "hasar", "kayıp", "eksik", "kırık", "patlak", "ezik", "çözüm", "memnun",
            "logistic", "transport", "transit", "dispatch", "ship", "track", "distribut", "branch", 
            "warehouse", "inventor", "suppl", "bill", "invoic", "receipt", "complain", "request", 
            "help", "call", "contact", "communic", "authoriz", "expert", "technic", "repair", "fix", 
            "install", "assembl", "bag", "late", "delay", "damag", "lost", "miss", "broken", "solut", "satisf"],
            




            "Ekonomi / Fiyat (Economy / Price)": ["fiyat", "ücret", "para", "maliyet", "indirim", "kampanya", "tutar", "fatura", 
            "ödeme", "taksit", "vergi", "zam", "ucuzluk", "pahalı", "maaş", "price", "cost", "money", "discount", "campaign", "amount",
            "bill", "payment", "installment", "tax", "raise", "cheap", "expensive", "salary","ekonom", "finans", "banka", "kart", 
            "kredi", "faiz", "borç", "bütçe", "cüzdan", "nakit","bozukluk", "komisyon", "kesinti", "hesap", "limit", "yatırım", "borsa",
            "kripto", "coin","hisse", "fon", "döviz", "kur", "dolar", "euro", "altın", "piyasa", "enflasyon", "zarar", 
            "kar ", "kazanç", "gelir", "gider", "bedel", "eder", "ucuz", "kazık", "soygun", "makbuz",
            "havale", "eft", "iban", "harcama", "birikim", "mevduat", "kasiyer", "vezne", "ödül",
            "econom", "financ", "bank", "card", "credit", "interest", "debt", "budget", "wallet", 
            "cash", "coin", "commis", "deduct", "account", "limit", "invest", "stock", "crypto", 
            "fund", "currency", "rate", "dollar", "euro", "gold", "market", "inflat", "loss", 
            "profit", "earn", "income", "expens", "value", "worth", "cheap", "scam", "receipt", 
            "transfer", "spend", "saving", "deposit", "cashier", "reward"],



            
            "Gıda / Restoran (Food / Dining)": ["yemek", "lezzet", "tat", "porsiyon", "ikram", "menü", "sıcaklık", "tazelik", "içecek",
            "kahve", "tatlı", "çorba", "sos", "food", "taste", "flavor", "portion", "menu", "temperature", "freshness", "drink", "beverage", 
            "coffee", "sweet", "dessert", "soup", "sauce","garson", "şef", "aşçı", "restoran", "lokanta", "kafe", "cafe", "masa", "hesap", "adisyon", 
            "bahşiş", "et", "tavuk", "balık", "döner", "kebap", "burger", "pizza", "makarna", "salata", 
            "sebze", "meyve", "ekmek", "peynir", "yağ", "tuz", "baharat", "şeker", "çay", "su", "ayran", 
            "kola", "içki", "şarap", "bira", "sıcak", "soğuk", "taze", "bayat", "çiğ", "yanık", "piş", 
            "leziz", "enfes", "gurme", "doyurucu", "hijyen", "temiz", "pis", "kıl", "tüy", "böcek", "sinek",
            "vegan", "vejetaryen", "organik", "tarif", "mutfak", "öğün", "kahvalt", "akşam yem", "öğle yem",
            "waiter", "waitress", "server", "chef", "cook", "restaur", "cafe", "diner", "table", "bill", 
            "check", "tip", "meat", "chicken", "fish", "beef", "pork", "burger", "pizza", "pasta", 
            "salad", "veg", "fruit", "bread", "cheese", "oil", "butter", "salt", "spice", "sugar", 
            "tea", "water", "soda", "coke", "juice", "wine", "beer", "alcohol", "hot", "cold", "fresh", 
            "stale", "raw", "burnt", "delici", "yum", "gourmet", "fill", "hygien", "clean", "dirt", 
            "bug", "hair", "fly", "vegan", "organic", "recipe", "kitchen", "meal", "dine", "dining",
            "breakfast", "lunch", "dinner"],



            
            "Mekan / Atmosfer (Location / Atmosphere)": ["mekan", "ortam", "temizlik", "hijyen", "tuvalet", "bina", "oda", "otel", "sokak", 
            "şehir", "place", "venue", "atmosphere", "cleaning", "hygiene", "toilet", "building", "room", "hotel", "street", "city","salon", "teras", "balkon", "bahçe", "manzar", "dekor", "tasarım", "konsept", "havalandırm", 
            "klima", "ısıtma", "soğutma", "aydınlatma", "ışık", "müzik", "ses", "gürültü", "sessiz", 
            "kalabalık", "ferah", "basık", "dar", "geniş", "havasız", "pis", "koku", "kokuy", 
            "havuz", "lobi", "asansör", "merdiven", "otopark", "vale", "konum", "merkez", "ulaşım", 
            "mahalle", "cadde", "meydan", "zemin", "duvar", "tavan", "kapı", "pencere", "lavabo", "wc", 
            "banyo", "duş", "spa", "sauna", "spor salonu", "yatak", "çarşaf", "havlu", "manevi",
            "locat", "spaci", "cramp", "decor", "view", "light", "nois", "quiet", "crowd", "park", 
            "valet", "pool", "lobby", "elevat", "stair", "air cond", "ac ", "heat", "ventil", 
            "smell", "stink", "odor", "bathroom", "restroom", "washroom", "floor", "wall", "ceiling", 
            "door", "window", "neighborhood", "avenue", "center", "central", "transport", "shower", 
            "towel", "bed", "sheet", "ambianc", "vibe"],

        }

      

        # ULTRA-KAPSAYICI TÜRKÇE FİİL VE ŞAHIS EKLERİ FİLTRESİ
  
        fiil_ekleri = (
            # 1. Mastar ve İsim-Fiiller
            'mayı', 'meyi', 'mak', 'mek', 'maya', 'meye', 'ması', 'mesi',
            
            # 2. Şimdiki Zaman (-yor) ve Tüm Şahıs Ekleri
            'ıyorum', 'ıyorsun', 'ıyor', 'ıyoruz', 'ıyorsunuz', 'ıyorlar',
            'iyorum', 'iyorsun', 'iyor', 'iyoruz', 'iyorsunuz', 'iyorlar',
            'uyorum', 'uyorsun', 'uyor', 'uyoruz', 'uyorsunuz', 'uyorlar',
            'üyorum', 'üyorsun', 'üyor', 'üyoruz', 'üyorsunuz', 'üyorlar',
            'yorum', 'yorsun', 'yordu', 'iyordu', 'ıyordu', 'uyordu', 'üyordu', 'yormuş', 'yorsa',
            
            # 3. Gelecek Zaman (-ecek/-acak) ve Tüm Şahıs Ekleri
            'acağım', 'eceğim', 'acaksın', 'eceksin', 'acak', 'ecek', 
            'acağız', 'eceğiz', 'acaksınız', 'eceksiniz', 'acaklar', 'ecekler', 
            'acaktı', 'ecekti', 'acağı', 'eceği',
            
            # 4. Görülen Geçmiş Zaman (-di/-ti) ve Tüm Şahıs Ekleri
            'dım', 'dim', 'dum', 'düm', 'tım', 'tim', 'tum', 'tüm',
            'dın', 'din', 'dun', 'dün', 'tın', 'tin', 'tun', 'tün',
            'dı', 'di', 'du', 'dü', 'tı', 'ti', 'tu', 'tü',
            'dık', 'dik', 'duk', 'dük', 'tık', 'tik', 'tuk', 'tük',
            'dınız', 'diniz', 'dunuz', 'dünüz', 'tınız', 'tiniz', 'tunuz', 'tünüz',
            'dılar', 'diler', 'tılar', 'tiler',
            
            # 5. Öğrenilen Geçmiş Zaman (-miş) ve Tüm Şahıs Ekleri
            'mışım', 'mişim', 'muşum', 'müşüm', 
            'mışsın', 'mişsin', 'muşsun', 'müşsün', 
            'mış', 'miş', 'muş', 'müş',
            'mışız', 'mişiz', 'muşuz', 'müşüz', 
            'mışsınız', 'mişsiniz', 'muşsunuz', 'müşsünüz', 
            'mışlar', 'mişler', 'muşlar', 'müşler',
            'mıştı', 'mişti', 'muştu', 'müştü',
            
            # 6. Gereklilik Kipi (-malı/-meli) ve Tüm Şahıs Ekleri
            'malıyım', 'meliyim', 'malısın', 'melisin', 'malı', 'meli',
            'malıyız', 'meliyiz', 'malısınız', 'melisiniz', 'malılar', 'meliler', 
            'malıydı', 'meliydi',
            
            # 7. Şart Kipi (-sa/-se) ve Şahıs Ekleri
            'sam', 'sem', 'san', 'sen', 'sak', 'sek', 
            'sanız', 'seniz', 'salar', 'seler',
            'saydım', 'seydim', 'saydın', 'seydin', 'saydı', 'seydi', 'saydık', 'seydik',
            
            # 8. İstek Kipi (-a/-e) ve Şahıs Ekleri
            'ayım', 'eyim', 'alım', 'elim', 'asın', 'esin', 'asınız', 'esiniz',
            
            # 9. Zarf-Fiil (Zaman ve Durum bildiren bağlaç ekleri)
            'arak', 'erek', 'ınca', 'ince', 'unca', 'ünce', 
            'dıkça', 'dikçe', 'tıkça', 'tikçe', 
            'madan', 'meden', 'ken', 'alı', 'eli',
            'dığında', 'diğinde', 'duğunda', 'düğünde', 'tığında', 'tiğinde', 'tuğunda', 'tüğünde',
            'dığımda', 'diğimde', 'duğumda', 'düğümde', 'tığımda', 'tiğimde', 'tuğumda', 'tüğümde',
            
            # 10. İngilizce Fiil Ekleri
            'ing','ed','ied','izing','ising','ized','ised','ating','ated','ifying','ified'
        )
  # 
  # 3. ANALİZ DÖNGÜSÜ
        for parca in parcalar:
            # =================================================================
            # A) Duyguyu Bul
            # =================================================================
            girdiler = tokenizer(parca, return_tensors="pt", truncation=True, padding=True, max_length=128)
            with torch.no_grad(): 
                ciktilar = model(**girdiler)
                T_parca = 0.7                                      
                olasiliklar = torch.softmax(ciktilar.logits / T_parca, dim=-1)
                p_neg = olasiliklar[0][0].item()
                p_pos = olasiliklar[0][1].item()
                
                yogunluk_matematigi = p_pos - p_neg
                
                # Parçalar için NÖTR EŞİĞİ!
                if abs(yogunluk_matematigi) < 0.20:
                    duygu = "Nötr"
                    skor_float = 0.50 # Oranlar eşit olduğu için güveni dengede tutar
                elif p_pos > p_neg:
                    duygu = "Olumlu"
                    skor_float = p_pos
                else:
                    duygu = "Olumsuz"
                    skor_float = p_neg
                    
                guven_skoru = f"%{skor_float * 100:.1f}"
                duygu_yogunlugu = f"{yogunluk_matematigi:+.2f}"

            # =================================================================
            # B) Nedeni/Sıfatı Bul
            # =================================================================
            derece_zarflari = [
                'çok', 'aşırı', 'oldukça', 'gayet', 'fazla', 'baya', 'bayağı', 'daha', 'en', 
                'inanılmaz', 'gerçekten', 'süper', 'az', 'biraz', 'tamamen', 'resmen', 'adeta', 
                'epey', 'hayli', 'müthiş', 'fena', 'son derece', 'korkunç', 'ekstra',
                'very', 'too', 'so', 'really', 'extremely', 'quite', 'highly', 'incredibly', 'totally'
            ]
            
            # ORTAK KALKAN: Hem hedefler hem sebepler için kullanılacak tek ve güçlü liste!
            YASAKLI_KELIMELER = set([
                "pek", "çok", "aşırı", "oldukça", "gayet", "fazla", "baya", "bayağı", "daha", "en",
                "az", "biraz", "tamamen", "epey", "hayli", "hiç", "inanılmaz", "gerçekten",
                "ve", "veya", "ya", "da", "de", "ile", "ki", "ama", "fakat", "lakin", "ancak",
                "yalnız", "oysa", "oysaki", "halbuki", "çünkü", "zira", "madem", "mademki",
                "meğer", "meğerse", "hatta", "üstelik", "ayrıca", "hem", "ise", "yine", "yinede",
                "ister", "yahut", "veyahut", "nede", "ne", "bile", "dahi", "için", "gibi", "kadar",
                "göre", "rağmen", "diye", "üzere", "dolayı", "yüzden", "sebeple", "nedenle",
                "bu yüzden", "bundan dolayı", "bu sebeple", "şu sebeple", "o sebeple", 
                "sonuç olarak", "netice itibariyle", "bu durumda", "şu durumda",
                "bununla birlikte", "bununla beraber", "ne var ki", "buna rağmen", 
                "diğer yandan", "öte yandan", "tam tersine", "bunun aksine", "hal böyleyken",
                "ek olarak", "bunun dışında", "bunun haricinde", "ne yazık ki", 
                "her halükarda", "uzun lafın kısası", "sözün kısası", "bir de",
                "because of", "due to", "owing to", "as a result", "for this reason", 
                "as a consequence","even though", "even if", "on the contrary", "on the other hand", 
                "in contrast", "despite this", "in spite of", "having said that",
                "in addition", "as well as", "apart from", "aside from", "what is more", 
                "provided that", "as long as", "in case", "up to"])
            
            bulunan_sebepler = []
            # --- YENİ EKLENEN BÜTÜNSEL SİLİCİ ---
            gecici_parca = parca.lower()
            for yasakli in YASAKLI_KELIMELER:
                if " " in yasakli:  # Listedeki kelimenin içinde boşluk varsa (Yani "bu yüzden" gibi bir öbekse)
                    # O öbeği cümleden bütün olarak kazı ve sil
                    gecici_parca = re.sub(rf'\b{yasakli}\b', '', gecici_parca)
            
            parca_kelimeleri = gecici_parca.split()
            # ------------------------------------
            kullanilan_kokler = [] 
            
            pozitif_puan = 0
            negatif_puan = 0
            
            for idx, kelime in enumerate(parca_kelimeleri):
                arama_k = re.sub(r'[^\w\s]', '', kelime.lower()) 
                if len(arama_k) < 2: continue

                if 'kesin_copluk' in locals() and arama_k in kesin_copluk: 
                    continue

                # EĞER ZARF VEYA BAĞLAÇSA HİÇ SEBEP ARAMA, GEÇ!
                if arama_k in YASAKLI_KELIMELER:
                    continue

                bulundu = False
                # 1. Pozitiflerde ara
                for ifade in pozitif_ifadeler:
                    if arama_k.startswith(ifade):
                        sebep_grubu = arama_k
                        # (Mevcut Zarf Kontrolün)
                        if idx > 0:
                            onceki_kelime = re.sub(r'[^\w\s]', '', parca_kelimeleri[idx-1].lower())
                            if onceki_kelime in derece_zarflari:
                                sebep_grubu = f"{onceki_kelime} {arama_k}"
                        
                        #  YENİ EKLENEN: GENİŞLETİLMİŞ KAPSAYICI NEGASYON (3 KELİME UZAĞA BAKAR)
                        yon_degisti = False
                        
                        # 1. TÜRKÇE MANTIĞI: Sağdaki 3 kelimeye kadar "değil", "yok", "hiç", "asla" taraması yap
                        tr_negations = ['değil', 'yok', 'hiç', 'asla', 'imkansız', 'sıfır','kesinlikle']
                        for i in range(1, 4): # 1, 2 ve 3 kelime sonrasına teker teker bak
                            if idx + i < len(parca_kelimeleri):
                                sonraki_k = re.sub(r'[^\w\s]', '', parca_kelimeleri[idx+i].lower())
                                # Eğer kelime listemizdeki herhangi bir olumsuzlukla başlıyorsa
                                if any(sonraki_k.startswith(neg) for neg in tr_negations):
                                    sebep_grubu = f"{sebep_grubu} {sonraki_k}" # Örn: "iyi" ve "değil" birleşir
                                    kullanilan_kokler.append(sonraki_k) 
                                    yon_degisti = True 
                                    break # Olumsuzluğu bulduk, daha uzağa bakmaya gerek yok!

                        # 2. İNGİLİZCE MANTIĞI: Soldaki 3 kelimeye kadar "not", "never", "n't" vb. taraması yap
                        en_negations = ["not", "never", "no", "nobody", "nothing", "hardly", "barely", "dont", "doesnt", "didnt", "isnt", "arent", "wasnt", "werent", "wont", "cant", "couldnt", "shouldnt", "wouldnt", "hasnt", "havent", "hadnt"]
                        if not yon_degisti: # Türkçe veto çalışmadıysa sola (İngilizce'ye) bak
                            for i in range(1, 4): # 1, 2 ve 3 kelime öncesine teker teker bak
                                if idx - i >= 0:
                                    onceki_k = re.sub(r'[^\w\s]', '', parca_kelimeleri[idx-i].lower())
                                    if onceki_k in en_negations or onceki_k.endswith("nt"):
                                        sebep_grubu = f"{onceki_k} {sebep_grubu}" # Örn: "not" ve "very happy" birleşir
                                        kullanilan_kokler.append(onceki_k)
                                        yon_degisti = True
                                        break

                        if sebep_grubu not in bulunan_sebepler:
                            bulunan_sebepler.append(sebep_grubu)
                            kullanilan_kokler.append(arama_k)
                            
                            #  YENİ EKLENEN PUANLAMA MANTIĞI: Yön değiştiyse eksiye puan yaz!
                            if yon_degisti:
                                negatif_puan += 1
                            else:
                                pozitif_puan += 1
                                
                            if idx > 0 and onceki_kelime in derece_zarflari:
                                kullanilan_kokler.append(onceki_kelime)
                        bulundu = True
                        break
                
                if bulundu: continue
                
                # 2. Negatiflerde ara
                for ifade in negatif_ifadeler:
                    if arama_k.startswith(ifade):
                        sebep_grubu = arama_k
                        # (Mevcut Zarf Kontrolün)
                        if idx > 0:
                            onceki_kelime = re.sub(r'[^\w\s]', '', parca_kelimeleri[idx-1].lower())
                            if onceki_kelime in derece_zarflari:
                                sebep_grubu = f"{onceki_kelime} {arama_k}"
                        
                        #  YENİ EKLENEN: Sağdaki kelime "değil" veya "yok" köküyle mi başlıyor?
                        yon_degisti = False
                        if idx + 1 < len(parca_kelimeleri):
                            sonraki_k = re.sub(r'[^\w\s]', '', parca_kelimeleri[idx+1].lower())
                            # startswith ile ekleri de yakalar
                            if sonraki_k.startswith('değil') or sonraki_k.startswith('yok'):
                                sebep_grubu = f"{sebep_grubu} {sonraki_k}" 
                                kullanilan_kokler.append(sonraki_k)
                                yon_degisti = True # Alarm: Duygu tersine döndü!

                        if sebep_grubu not in bulunan_sebepler:
                            bulunan_sebepler.append(sebep_grubu)
                            kullanilan_kokler.append(arama_k)
                            
                            #  YENİ EKLENEN PUANLAMA MANTIĞI: Yön değiştiyse artıya puan yaz!
                            if yon_degisti:
                                pozitif_puan += 1
                            else:
                                negatif_puan += 1
                                
                            if idx > 0 and onceki_kelime in derece_zarflari:
                                kullanilan_kokler.append(onceki_kelime) 
                        break

            temiz_sebepler = []
            for sebep in bulunan_sebepler:
                alt_kume_mi = any(sebep != diger and sebep in diger for diger in bulunan_sebepler)
                if not alt_kume_mi:
                    temiz_sebepler.append(sebep)
            
            sebep_kelimesi = ", ".join(temiz_sebepler) if temiz_sebepler else "belirtilmemiş"

            # -----------------------------------------------------------------
            #  YAPAY ZEKA VETOSU (LEXICON OVERRIDE)
            # -----------------------------------------------------------------
            if pozitif_puan > negatif_puan and duygu == "Olumsuz":
                duygu = "Olumlu"
                duygu_yogunlugu = f"+{abs(float(duygu_yogunlugu)):.2f}"
                guven_skoru = "%99.9"
            elif negatif_puan > pozitif_puan and duygu == "Olumlu":
                duygu = "Olumsuz"
                duygu_yogunlugu = f"-{abs(float(duygu_yogunlugu)):.2f}"
                guven_skoru = "%99.9"

            # =================================================================
            # C) Hedefi/Özneyi Bul
            # =================================================================
            #  KAPSAMLI FİİL KÖKÜ ÇÖPLÜĞÜ 
            # (Ekler zaten 'fiil_ekleri' filtresine takılacağı için burada sadece kökler var)
            YASAKLI_FIILLER = set([
                # Türkçe Sadece Fiil Kökleri
                "yap", "et", "al", "ver", "git", "gel", "kullan", "çalış", "oyna", "bak", 
                "gör", "izle", "duy", "dinle", "ye", "iç", "söyle", "konuş", "yaz", "oku", 
                "düşün", "bekle", "başla", "bit", "geç", "kal", "çek", "at", "vur", "kır", 
                "boz", "aç", "kapat", "koy", "sev", "beğen", "iste", "ol", "öl", "getir", 
                "götür", "taşı", "çık", "in", "düş", "kalk", "otur", "dur", "sor",
                "cevapla", "ara", "bul", "bil", "tanı", "anla", "anlat", "öde", "sat",
                
                # İngilizce Yalın Fiiller ve Düzensiz Fiiller (Irregular Verbs)
                # Not: 'ed', 'ing' takısı alan düzenli fiiller filtreye takılır, buraya gerek yok.
                "do", "did", "done", "make", "made", "get", "got", "go", "went", "gone", 
                "come", "came", "take", "took", "taken", "give", "gave", "given", "use", 
                "work", "play", "look", "see", "saw", "seen", "watch", "eat", "ate", "eaten", 
                "drink", "drank", "say", "said", "tell", "told", "speak", "spoke", "spoken", 
                "write", "wrote", "written", "read", "think", "thought", "wait", "start", 
                "stop", "finish", "open", "close", "love", "like", "hate", "want", "need", 
                "buy", "bought", "sell", "sold", "pay", "paid", "try", "feel", "felt", 
                "find", "found", "leave", "left", "bring", "brought", "keep", "kept", 
                "hold", "held", "know", "knew", "known", "catch", "caught"
            ])
            bulunan_hedef = "Belirtilmemiş"
            if len(parca_kelimeleri) > 0:
                guclu_isimler = []
                yedek_isimler = []
                
                for k in parca_kelimeleri:
                    temiz_k = k.lower()
                    temizlenmis_isim = re.sub(r'[^\w\s]', '', temiz_k)
                    
                    # 1. ve 2. Filtre: Yasaklıysa veya Kök olarak kullanıldıysa at!
                    if temizlenmis_isim in YASAKLI_KELIMELER: continue
                    if temizlenmis_isim in kullanilan_kokler: continue
                    if temizlenmis_isim in YASAKLI_FIILLER: continue
                    
                    # 3. ve 4. Filtre: Çöplükteyse veya fiil ise at!
                    try:
                        if 'kesin_copluk' in locals() and temiz_k in kesin_copluk: continue
                        if 'fiil_ekleri' in locals() and len(temiz_k) > 4 and temiz_k.endswith(fiil_ekleri): continue
                    except: pass
                        
                    if len(temizlenmis_isim) > 1:
                        try:
                            if 'zayif_isimler' in locals() and temiz_k in zayif_isimler:
                                yedek_isimler.append(temizlenmis_isim)
                            else:
                                guclu_isimler.append(temizlenmis_isim)
                        except:
                            guclu_isimler.append(temizlenmis_isim)
                
                if len(guclu_isimler) > 1:
                    isim1, isim2 = guclu_isimler[0], guclu_isimler[1]
                    try:
                        kelimeler_kucuk = [kel.lower() for kel in parca_kelimeleri]
                        index1 = next(i for i, kel in enumerate(kelimeler_kucuk) if isim1 in kel)
                        index2 = next(i for i, kel in enumerate(kelimeler_kucuk) if isim2 in kel)
                        if abs(index1 - index2) == 1:
                            bulunan_hedef = f"{isim1} {isim2}"
                        else:
                            bulunan_hedef = isim1
                    except: bulunan_hedef = isim1
                elif len(guclu_isimler) == 1:
                    bulunan_hedef = guclu_isimler[0] 
                elif len(yedek_isimler) > 0:
                    bulunan_hedef = yedek_isimler[0]
                else:
                    # AKILLI KURTARICI: Cümlede "arabam" varsa ve fiil sanılıp elendiyse onu geri getir.
                    for kelime in parca_kelimeleri:
                        k_temiz = re.sub(r'[^\w\s]', '', kelime.lower())
                        # Eğer kelime yasaklı bir kelime ("bu", "şu", "pek") DEĞİLSE mecburen hedef yap
                        if k_temiz and k_temiz not in YASAKLI_KELIMELER and k_temiz not in kullanilan_kokler and k_temiz not in YASAKLI_FIILLER:
                            try:
                                if 'fiil_ekleri' in locals() and len(k_temiz) > 4 and k_temiz.endswith(fiil_ekleri): continue
                            except: pass
                            bulunan_hedef = k_temiz
                            break

            # =================================================================
            # ÇOKLU KATEGORİ ÇIKARIMI
            # =================================================================
            bulunan_kategoriler = [] 
            hedef_kelimeler = bulunan_hedef.lower().split()
            
            for kelime in hedef_kelimeler:
                try:
                    if 'kategori_sozlugu' in locals():
                        for kategori_adi, anahtar_kelimeler in kategori_sozlugu.items():
                            for anahtar in anahtar_kelimeler:
                                if kelime == anahtar: 
                                    if kategori_adi not in bulunan_kategoriler:
                                        bulunan_kategoriler.append(kategori_adi)
                                    break
                except: pass
            
            if len(bulunan_kategoriler) == 0:
                bulunan_kategori = "Genel / Diğer"
            else:
                bulunan_kategori = " & ".join(bulunan_kategoriler) 

            # =================================================================
            # D) SONUÇLARI KAYDET
            # =================================================================
            detayli_sonuclar.append({
                'parca_metni': parca,
                'hedef': bulunan_hedef,
                'kategori': bulunan_kategori,
                'duygu': duygu,
                'guven_skoru': guven_skoru,
                'duygu_yogunlugu': duygu_yogunlugu, 
                'sebep_eylem': sebep_kelimesi
            })
            # =================================================================
        # YENİ EKLENEN: MATEMATİKSEL HOLİSTİK (GENEL) SONUÇ HESAPLAMA
        # =================================================================
        toplam_skor = 0.0
        
        for sonuc in detayli_sonuclar:
            # "+0.95" veya "-0.80" gibi olan parça yoğunluklarını topla
            toplam_skor += float(sonuc['duygu_yogunlugu'])
            
        # Parça sayısına bölerek net ortalamayı bul (Örn: +0.90 ve -0.85 gelirse ortalama +0.025 olur)
        ortalama_yogunluk = toplam_skor / len(detayli_sonuclar) if detayli_sonuclar else 0.0
        
        # Ortalama skoru %0 ile %100 arasına çek (Gerçek oranlar için)
        p_pos_genel = (ortalama_yogunluk + 1) / 2 
        p_neg_genel = 1 - p_pos_genel
        
        genel_olumlu_oran = p_pos_genel * 100
        genel_olumsuz_oran = p_neg_genel * 100

        # NÖTR EŞİĞİ (Fark 0.20'den küçükse, yani cümleler arası çelişki varsa)
        if abs(ortalama_yogunluk) < 0.20:
            genel_karar = "NÖTR"
            yogunluk_yonu = "nötr (dengeli)"
        elif ortalama_yogunluk > 0:
            genel_karar = "OLUMLU"
            yogunluk_yonu = "olumlu"
        else:
            genel_karar = "OLUMSUZ"
            yogunluk_yonu = "olumsuz"
            
        total_aciklama = f"Bu ifade genel olarak %{genel_olumlu_oran:.1f} olumlu, %{genel_olumsuz_oran:.1f} olumsuz ve {ortalama_yogunluk:+.2f} duygu yoğunluğu ortalamasıyla {yogunluk_yonu}. Yani bu cümlenin genel duygusu {genel_karar}DUR."
          
          
            # =================================================================
        # E) OKUNABİLİR "GENEL ÖZET" ÜRETİCİSİ (Evrensel NLG Modülü)
        # =================================================================
        genel_ozet = ""
        ozet_parcalari = []
        
        for i, s in enumerate(detayli_sonuclar):
            durum = "olumlu bir yaklaşım sergilenmiş" if s['duygu'] == "Olumlu" else "olumsuz bir görüş bildirilmiş"
            hedef_metni = s['hedef'] if s['hedef'] != "Belirtilmemiş" else "genel konu"
            sebep_metni = f"({s['sebep_eylem']} ifadesi kullanılarak) " if s['sebep_eylem'] != "belirtilmemiş" else ""
            
            cumlecik = f"'{hedef_metni}' hakkında {sebep_metni}{durum}"
            
            if i > 0:
                onceki_duygu = detayli_sonuclar[i-1]['duygu']
                if s['duygu'] != onceki_duygu:
                    cumlecik = "ancak " + cumlecik 
                else:
                    cumlecik = "ve ayrıca " + cumlecik 
            
            ozet_parcalari.append(cumlecik)
            
        if ozet_parcalari:
            genel_ozet = "Analiz özetine göre; " + ", ".join(ozet_parcalari) + "."
            genel_ozet = genel_ozet.replace("  ", " ") 
        else:
            genel_ozet = "Metinden belirgin bir özet çıkarılamadı."

        return jsonify({
            'orijinal_metin': metin,
            'total_aciklama': total_aciklama,
            'genel_ozet': genel_ozet, # <-- ÖZETİ BURAYA EKLEDİK
            'analizler': detayli_sonuclar
        })

    except Exception as e:
        return jsonify({'hata': str(e)}), 500
    

# (DOKUNMADIK) ESKİ TOPLU TAHMİN MOTORU (Artık Akıllı!)
@app.route('/toplu-analiz', methods=['POST'])
def toplu_analiz():
    try:
        if 'dosya' not in request.files: return jsonify({'hata': 'Lütfen bir CSV dosyası yükleyin.'}), 400
        dosya = request.files['dosya']
        df = pd.read_csv(dosya)
        
        metin_sutunu = next((kolon for kolon in df.columns if kolon.lower() in ['metin', 'yorum', 'comment', 'review']), None)
        if not metin_sutunu: return jsonify({'hata': 'Excel/CSV dosyasında geçerli sütun bulunamadı!'}), 400
        
        metinler = df[metin_sutunu].dropna().astype(str).tolist() 
        olumlu_metinler, olumsuz_metinler = [], []
        
        for metin in metinler:
            girdiler = tokenizer(metin, return_tensors="pt", truncation=True, padding=True, max_length=128)
            with torch.no_grad(): sinif = torch.argmax(model(**girdiler).logits, dim=-1).item()
            if sinif == 1: olumlu_metinler.append(metin)
            else: olumsuz_metinler.append(metin)
                
        toplam = len(metinler)
        olumlu_yuzde = (len(olumlu_metinler) / toplam) * 100 if toplam > 0 else 0
        olumsuz_yuzde = (len(olumsuz_metinler) / toplam) * 100 if toplam > 0 else 0
        
        return jsonify({
         'toplam_yorum': toplam,
         'olumlu_orani': f"%{olumlu_yuzde:.1f}",
         'olumsuz_orani': f"%{olumsuz_yuzde:.1f}",
         # İŞTE BURASI DEĞİŞTİ: Artık yeni dinamik fonksiyonumuzu çağırıyoruz!
         'olumlu_sebepler': dinamik_neden_bul(olumlu_metinler), 
         'olumsuz_sebepler': dinamik_neden_bul(olumsuz_metinler)
        })
    except Exception as e:
        return jsonify({'hata': f"Dosya okuma hatası: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)

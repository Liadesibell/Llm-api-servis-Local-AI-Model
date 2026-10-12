# Ink AI

Ink AI, dil modelini kendi bilgisayarında çalıştıran yerel yapay zekâ
yardımcısıdır. Bu API model sağlayıcısına veya Ollama'ya bağlanmaz; GGUF
dosyasını doğrudan `llama-cpp-python` ile açar.

## Gerekenler

- Windows 10 veya 11
- Python 3.12 veya üstü
- Proje bağımlılıkları kurulu sanal ortam
- `models` klasöründe indirme komutuyla kurulan GGUF modelleri

Coder ve Reasoning rolleri varsayılan olarak `models/qwen3.5-35b-a3b-q4_k_m.gguf`
dosyasındaki Qwen3.5-35B-A3B Q4_K_M modelini paylaşır. Model yaklaşık 22 GB'dır;
indirme komutu eksik modelleri kurar.

## API'yi başlatma

Proje klasöründe PowerShell açın; ilk çalıştırmada ortamı eşitleyin:

```powershell
uv sync
```

Ardından şu komutla API'yi başlatın:

```powershell
uv run ink-ai
```

API yalnızca bu bilgisayardan erişilebilen `127.0.0.1:8000` adresinde açılır.
Model dosyası bulunamazsa API yine başlar; sağlık uç noktası bunu bildirir ve
sohbet isteği `503` hatası verir.

## Masaüstü arayüzünü başlatma

Bağımlılıkları eşitleyin ve API'yi bir PowerShell penceresinde başlatın:

```powershell
uv sync
uv run ink-ai
```

Ardından ikinci bir PowerShell penceresinde sohbet arayüzünü açın:

```powershell
uv run ink-ai-gui
```

VS Code içindeki importlar sarı çizgiliyse `Ctrl+Shift+P` → **Python: Select
Interpreter** komutundan proje içindeki `.venv\Scripts\python.exe` yorumlayıcısını
seçin ve **Developer: Reload Window** ile pencereyi yenileyin. Paketleri
`pip` ile başka Python kurulumuna değil, proje ortamına `uv sync` ile kurun.

Arayüz varsayılan olarak `http://127.0.0.1:8000/api/sohbet` adresine bağlanır.
Farklı bir API adresi kullanıyorsanız sol paneldeki **API adresi** alanını
güncelleyin. Üstteki model menüsünden otomatik seçim veya Flash, Reasoning ve
Coder uzman modellerinden birini seçebilirsiniz; otomatik seçim belirgin
matematik ve kod isteklerini ilgili uzman modele doğrudan yönlendirir. Yeni
sohbet düğmesi ekrandaki konuşmayı temizler; API isteği yerel sunucuya gönderilir.
Üst sağdaki görünüm menüsünden beyaz/açık veya neon siyah/koyu temayı seçin;
tercih `~/.ink-ai/gui_preferences.json` dosyasında saklanır. Yeşil vurgular
parlamayan, yumuşak tondadır. Sohbet yanıtları model ürettikçe akış halinde
gösterilir; modelin düşünme/yönlendirme ve araştırma aşamalarında durum bilgisi
görünür. Kod blokları koyu kod alanında, terminal komutları Linux terminali
görünümünde ve terminale özgü yazım hızıyla biçimlendirilir. Akış desteği
olmayan istemciler için `/api/sohbet` varsayılan olarak normal JSON yanıtını
korur; GUI akışlı yanıtı kendisi ister.

Sohbet geçmişi bilgisayarda `~/.ink-ai/conversations.sqlite3` dosyasında
saklanır; istenirse `INK_AI_DATA_DIR` ortam değişkeniyle veri klasörü
değiştirilebilir. Uygulama yeniden açıldığında en son sohbeti otomatik yükler;
akış kesilse bile üretilmiş kısmi yanıt saklanır. Mesajlardaki **Kopyala**
düğmesi kod dâhil tüm mesajı panoya kopyalar. Ataç
düğmesi `.py`, `.txt`, `.md`, `.json` gibi metin/kod dosyalarını, PDF ve Word
`.docx` belgelerini, PNG/JPEG/WebP görsellerini ekler. Pano kopyalanmış görseli
veya PDF dosyasını yazma alanında `Ctrl+V` ile ekleyebilirsin. Belgeler yerel
API'ye gönderilir; en fazla 5 belge ve toplam 6.000 karakter desteklenir.
Görseller sohbet geçmişinde saklanır; en fazla 5 görsel, görsel başına 5 MB ve
toplam 10 MB sınırı vardır. Taranmış, metin katmanı olmayan PDF'lerden metin çıkarılamaz. Mevcut
sohbet modelleri metin tabanlıdır; API görsel eklerini kabul eder fakat görsel
içeriğini yorumlamaz ve bu sınırı asistana bildirir.

## Uç noktaları deneme

Sağlık kontrolü:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/saglik
```

Sohbet:

```powershell
$istek = @{
    mesaj = "Python'da liste nedir?"
    gecmis = @()
} | ConvertTo-Json

Invoke-RestMethod `
    -Uri http://127.0.0.1:8000/api/sohbet `
    -Method Post `
    -ContentType "application/json" `
    -Body $istek
```

Sohbet isteğinin `gecmis` alanı isteğe bağlıdır. Önceki mesajlar
`{"rol":"kullanici","icerik":"..."}` veya
`{"rol":"asistan","icerik":"..."}` biçiminde gönderilir. Bir istekte en fazla
20 geçmiş mesaj bulunabilir.

## Uç noktalar

| Yöntem | Yol | Açıklama |
| --- | --- | --- |
| `GET` | `/` | API'nin çalıştığını bildirir |
| `GET` | `/api/saglik` | Modelin yüklenip yüklenmediğini bildirir |
| `GET` | `/api/modeller` | Yerel metin modellerinin durumunu listeler |
| `GET` | `/api/yetenekler` | Otomatik yönlendirme ve API sınırlarını bildirir |
| `GET` | `/api/bilgi/durumu` | Yerel bilgi indeksinin klasör ve istatistiklerini verir |
| `POST` | `/api/bilgi/indeksle` | Bilgi klasörünü yeniden tarar ve indeksi günceller |
| `GET` | `/api/bilgi/ara?q=...` | Yerel BM25 aramasıyla kaynak parçalarını döndürür |
| `POST` | `/api/bilgi/metin` | Metni doğrudan yerel indekse ekler |
| `POST` | `/api/arastir` | DuckDuckGo'da ara, sayfaları oku ve kaynakları indekse ekle |
| `POST` | `/api/sohbet` | İsteğe uygun yerel uzman modeli otomatik seçip yanıt üretir |
| `POST` | `/api/sohbet/json` | Sohbet yanıtını doğrulanmış JSON nesnesi olarak döndürür |
| `POST` | `/api/belge/analiz` | Belge özetleme, belgeye dayalı soru-cevap ve alan çıkarımı yapar |
| `POST` | `/api/kod/analiz` | Kod inceleme/açıklama/hata ayıklama veya log tanılama yapar |
| `POST` | `/api/gorsel` | Yerel Stable Diffusion ile görsel üretir |
| `GET` | `/api/gorsel/{dosya}` | Üretilen yerel görseli döndürür |
| `GET` | `/v1/models` | Yerel ve yapılandırılmış harici model kataloğunu OpenAI biçiminde verir |
| `GET` | `/v1/providers` | Ollama/OpenAI-uyumlu sağlayıcı durumunu bildirir |
| `POST` | `/v1/chat/completions` | OpenAI uyumlu sohbet/akış ve güvenli araç önerileri |
| `POST` | `/v1/embeddings` | Yapılandırılmış uzak sağlayıcıdan embedding üretir |
| `POST` | `/v1/images/generations` | Görsel üretimini OpenAI Images yanıt biçiminde sunar |
| `GET/POST` | `/v1/collections` | İzole yerel bilgi koleksiyonlarını listeler/oluşturur |
| `POST` | `/v1/collections/{id}/documents` | Koleksiyona metin belgesi ekler |
| `GET` | `/v1/collections/{id}/search` | Yalnızca ilgili koleksiyonda BM25 araması yapar |
| `POST` | `/v1/evaluations` | En çok 10 örnekte model yanıtlarını ölçer |
| `POST` | `/v1/cancellations/{request_id}` | Aktif sohbet isteğini iptal etmeyi dener |
| `GET` | `/metrics` | Prometheus biçiminde istek/süre/hata sayaçlarını verir |

### OpenAI uyumlu model geçidi

`/v1/chat/completions` OpenAI sohbet sözleşmesini kullanır. `model` alanında
yerel `flash`, `reasoning`, `coder` modellerinden birini; Ollama için
`ollama/<kurulu-model>`; OpenAI veya başka bir OpenAI-uyumlu servis için
`openai/<model-id>` gönder. Yerel GGUF modelleri yoksa harici servisler yine
kullanılabilir. Yerel modelin desteklemediği parametre/özellikler için başarılı
ama etkisiz fallback yapılmaz; yanıt seçenekleri ve yetenekleri model
sağlayıcısına göre değişebilir.

Ollama için PowerShell ayarları:

```powershell
$env:INK_AI_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
uv run ink-ai
```

OpenAI API veya başka bir OpenAI-uyumlu sunucu için:

```powershell
$env:INK_AI_OPENAI_BASE_URL = "https://api.openai.com/v1"
$env:INK_AI_OPENAI_API_KEY = "API anahtarını güvenli ortam değişkeninden ayarla"
uv run ink-ai
```

Anahtar yalnızca API sunucusundan sağlayıcıya gider; model listesinde veya
istemci yanıtlarında döndürülmez. Kendi sunucunu kullanmak için base URL'yi
değiştirip sağlayıcının gerektirdiği anahtarı ayarla. İki sağlayıcı da aynı
anda yapılandırılabilir. Harici sağlayıcıların model listesi sağlayıcıdan canlı
alınır; sağlayıcı çevrimdışıysa `/v1/providers` hatayı bildirir.
`INK_AI_PROVIDER_TIMEOUT_SECONDS` uzak istek zaman aşımını (varsayılan 60,
1–300 aralığı) ayarlar.

```powershell
$body = @{
    model = "coder"
    messages = @(
        @{ role = "system"; content = "Kısa ve net yanıt ver." }
        @{ role = "user"; content = "FastAPI ne işe yarar?" }
    )
    max_tokens = 512
} | ConvertTo-Json -Depth 5

Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8000/v1/chat/completions `
  -ContentType "application/json" -Body $body
```

`stream = true` SSE akışı (`text/event-stream`) verir. Sağlayıcı destekliyorsa
akış uzak modelden aktarılır; yerel GGUF akışı da token geldikçe iletilir.
İstek `X-Request-ID` ile tanımlanabilir; aktif yerel üretimi ve uzak akışı
iptal etmek için `POST /v1/cancellations/{request_id}` çağrılabilir. İptal,
model sağlayıcısının iptal desteğine bağlıdır; normal, akışsız uzak sağlayıcı
HTTP isteği tamamlanana kadar iptal edilemez ve yapılandırılmış zaman aşımıyla
sınırlandırılır.
`response_format` için ortak yerel sözleşme `{"type":"json_object"}`'tir.
Embeddings için model kimliğini sağlayıcı önekiyle gönder; yerel GGUF sohbet
modellerinde embedding uç noktası bulunmadığından API açıkça `501` döndürür.
Yerel GGUF kullanım sayaçları tokenizer'dan değil metin uzunluğundan yaklaşık
hesaplanır ve yanıtta `usage_estimated: true` olarak işaretlenir; uzak sağlayıcı
kullanım sayaçları sağlayıcının döndürdüğü değerlerdir.

### Yerel bilgi koleksiyonları ve araç güvenliği

Koleksiyon oluşturup belge ekle:

```powershell
Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8000/v1/collections `
  -ContentType "application/json" -Body '{"id":"support"}'

$doc = @{
    name = "returns.md"
    content = "Refunds are available within 30 days."
} | ConvertTo-Json
Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8000/v1/collections/support/documents `
  -ContentType "application/json" -Body $doc
```

`/v1/chat/completions` isteğine `collection_id = "support"` eklenirse yerel
model yanıt bağlamında sadece bu koleksiyondaki eşleşmeleri kullanır.
Koleksiyonlar `INK_AI_COLLECTIONS_DIR` altında ayrı SQLite indeksleridir.
API anahtarı tek sunucu anahtarıdır; koleksiyon başına kullanıcı/rol ACL'si
henüz yoktur. Bu nedenle farklı müşterilere açılan bir serviste istemci başına
erişim denetimi eklemeden ortak API anahtarını paylaşma.

`tools` verildiğinde model fonksiyon çağrısı **önerisi** döndürebilir. API,
istemcinin fonksiyonunu çalıştırmaz. İstemci öneriyi kullanıcıya gösterip izin
aldıktan sonra kendi güvenli yürütme katmanında çalıştırmalı; model çıktısını
doğrudan komut/SQL/shell olarak yürütmemelidir.

### Güvenlik, limitler ve değerlendirme

`INK_AI_API_KEY` ayarlanırsa `/v1/*` uçları `Authorization: Bearer ...` veya
`X-API-Key` ister. Ayarlanmadığında mevcut yerel geliştirme davranışı korunur;
sunucu varsayılan olarak yalnızca `127.0.0.1:8000` üzerinde dinler. Rate limit
istemci IP'si başına bellekte uygulanır ve varsayılanı dakikada 60 `/v1`
isteğidir; `INK_AI_RATE_LIMIT_PER_MINUTE` ile 1–10.000 arasında ayarlanabilir.
Bu limit ve API anahtarı tek süreç içindir; çok worker/çok sunuculu kurulumda
paylaşılan bir rate-limit deposu ve kullanıcı bazlı kimlik doğrulama gerekir.
Her yanıt `X-Request-ID` içerir; istemci güvenli biçimde kendi kimliğini de
gönderebilir. `/metrics` genel istek sayısı, 5xx ve toplam süre sayaçlarını verir.

Model değerlendirmesi `/v1/evaluations` ile sınırlı sayıda prompt/anahtar ifade
örneğini çalıştırıp geçme oranı ve süre döndürür. Bu basit duman/regresyon
ölçümüdür; anlamsal kalite skoru değildir ve uzak sağlayıcı çağrısı maliyet
oluşturabilir.

```powershell
$evaluation = @{
    model = "coder"
    cases = @(
        @{
            prompt = "Python'da liste tipi nedir?"
            expected_contains = "liste"
        }
    )
} | ConvertTo-Json -Depth 5

Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8000/v1/evaluations `
  -ContentType "application/json" -Body $evaluation
```

### Belge analizi

En fazla beş belge ve toplam 6.000 karakter gönderilebilir. `task` değeri
`summarize`, `qa` veya `extract` olabilir. Soru-cevap için `question`, alan
çıkarımı için `fields` gereklidir:

```powershell
$body = @{
    task = "qa"
    question = "İade süresi kaç gün?"
    documents = @(@{
        name = "iade-politikasi.md"
        content = "İade talepleri satın alma tarihinden itibaren 30 gün içinde açılmalıdır."
    })
} | ConvertTo-Json -Depth 5

Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8000/api/belge/analiz `
  -ContentType "application/json" -Body $body
```

`extract` görevi çıkarılan alanları ayrıca JSON `result` alanında döndürür.
Belge içeriği yerel modele gönderilir; bu uç nokta dosyaları sunucuda saklamaz.

### Kod ve log analizi

`POST /api/kod/analiz` girdideki kodu veya logları çalıştırmadan inceler.
Kod için `review`, `debug` veya `explain`; log için `diagnose`, `debug` veya
`explain` görevlerinden birini kullan:

```powershell
$body = @{
    name = "app.log"
    kind = "logs"
    task = "diagnose"
    content = "ERROR database connection refused"
} | ConvertTo-Json

Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8000/api/kod/analiz `
  -ContentType "application/json" -Body $body
```

### JSON yanıt ve görsel üretimi

`POST /api/sohbet/json`, sohbet gövdesini (`mesaj`, `model`, `files` vb.)
normal sohbetle aynı biçimde alır. Model geçerli bir JSON nesnesi üretemezse
API düz metinle başarı dönmek yerine `502` verir. Akışlı sohbet bu uç noktada
desteklenmez. Görsel üretimi için `POST /api/gorsel` istem, adım sayısı ve
isteğe bağlı seed alır; yanıttaki `url` ile `GET /api/gorsel/{filename}`
üretim bilgisayarı üzerindeki görseli verir. Görsel modeli ve dosyalar yereldir.

## Yerel bilgi havuzu (RAG)

İndirip kullanım hakkına sahip olduğun teknik belgeleri `knowledge/incoming/`
klasörüne konuya göre alt klasörlerle koy:

```text
knowledge/incoming/
  python/
  powershell/
  docker/
  git/
  kubernetes/
  github-actions/
```

Desteklenen uzantılar `/api/yetenekler` yanıtında listelenir. PDF, DOCX,
Markdown, metin ve kod dosyalarındaki metin yerelde çıkarılır, yaklaşık 1.200
karakterlik örtüşen parçalara bölünür ve BM25 anahtar-kelime arama indeksi
SQLite'a yazılır. Görseller bilgi indeksi değildir; sohbetin görsel anlayışı da
sağlamaz. Taranmış PDF için OCR bu sürümde yoktur.

API çalışırken klasörü tekrar indeksle:

```powershell
Invoke-RestMethod -Method Post http://127.0.0.1:8000/api/bilgi/indeksle
Invoke-RestMethod http://127.0.0.1:8000/api/bilgi/durumu
Invoke-RestMethod 'http://127.0.0.1:8000/api/bilgi/ara?q=Docker%20Compose'
```

Sohbet isteğinde alakalı en fazla dört parça aranıp yerel modele bağlam olarak
verilir; kaynak kimlikleri yanıtın `X-Ink-AI-Sources` HTTP başlığında ve GUI'de
gösterilir. İndeks `%USERPROFILE%\.ink-ai\knowledge.sqlite3` içindedir.
Varsayılan klasör `knowledge/incoming`; `INK_AI_KNOWLEDGE_DIR` ve
`INK_AI_KNOWLEDGE_DB` ile konumları değiştirebilirsin. İçerik harici servise
gönderilmez. Yeniden indeksleme silinen dosyaların eski parçalarını temizler;
indekslemeden önce kaynak dosyaları sakla. PDF/DOCX ayrıştırma bağımlılıkları
gerekli olabilir. Bu arama semantik embedding değil, yerel BM25 tam/metin
arama yaklaşımıdır; Türkçe ekler ve anlam benzerliği nedeniyle ilgili olmayan
sonuçlar gelebilir. Yanıttaki kaynakları açıp doğrula.

Bir metni API'ye doğrudan eklemek için:

```powershell
$body = @{
    name = "deploy-notes.md"
    content = "Docker Compose servisleri compose.yaml dosyasında tanımlanır."
} | ConvertTo-Json
Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8000/api/bilgi/metin `
  -ContentType "application/json" -Body $body
```

## Coder LoRA ince ayarı (isteğe bağlı)

RAG dış kaynaklı bilgiyi bulup bağlama ekler; LoRA ise örneklerdeki yanıt
biçimini modele uydurmayı dener. İki işlem farklıdır. Mevcut `.gguf` dosyaları
eğitim girdisi değildir. Eğitim hattı Qwen2.5-Coder-7B-Instruct'in Hugging Face
temel ağırlıklarını kullanır; yeni Qwen3.5 GGUF ile aynı temel model değildir.
Hat Linux/WSL2 üzerinde 4-bit QLoRA adaptörü üretir ve mevcut GGUF'yi
kendiliğinden değiştirmez. Bu adaptörü Qwen3.5 modeline uygulamayın; temel model
eşleşmediğinden yükleme başarısız olur. 7B temel modelinin indirilmesi
gerekeceğinden diskte ek alan ve internet gerekir; 8 GB VRAM'de düşük batch ve
1.024 token bağlam kullanılır, yine de CUDA bellek yetersizliği olabilir.

Windows'ta WSL2 Ubuntu kullan. WSL2 içinde CUDA'nın `nvidia-smi` ile görünür
olduğunu doğrula, projeyi WSL dosya sistemine kopyala/clone et ve orada:

```bash
uv sync --group training
```

`training/raw/coder.jsonl` dosyasını kendin hazırlayıp örnekleri bir satırda bir
JSON nesnesi olarak yaz. Şema, sohbet biçimindedir:

```json
{"messages":[{"role":"user","content":"Bu Dockerfile'daki sorunu bul: ..."},{"role":"assistant","content":"Sorun ... Doğrulamak için ... komutunu çalıştır."}]}
```

Her örnekte en az bir `user` ve son olarak `assistant` mesajı bulunmalı.
Veri kümesini kişisel bilgi, API anahtarı, parola, lisanssız içerik, hatalı
yanıt ve test edilmemiş/yıkıcı komutlar açısından gözden geçir. En az 10 örnek
teknik alt sınırdır; işe yarar değerlendirme için yüzlerce kaliteli, çeşitli
örnek ve eğitimde kullanılmayan ayrı doğrulama soruları gerekir.

```bash
uv run python scripts/prepare_training_data.py training/raw/coder.jsonl
uv run python scripts/train_coder_lora.py
```

Adaptör `models/ink-ai-coder-lora/` altında oluşur. Bu başlangıç hattı eğitim
kayıtları/checkpoint üretir. API'nin kullanabilmesi için adaptörü, aynı temel
Hugging Face modeline karşılık gelen llama.cpp `convert_lora_to_gguf.py` aracıyla
GGUF-LoRA biçimine dönüştür:

```bash
python /path/to/llama.cpp/convert_lora_to_gguf.py \
  models/ink-ai-coder-lora \
  --base /path/to/Qwen2.5-Coder-7B-Instruct \
  --outfile models/ink-ai-coder-lora.gguf
```

Sonra API'yi başlatacağın WSL2 terminalinde adaptör yolunu belirt:

```bash
export INK_AI_CODER_LORA_PATH="$PWD/models/ink-ai-coder-lora.gguf"
uv run ink-ai
```

Uygulama bu adaptörü yalnız Coder GGUF modeline uygular; Flash/Reasoning ve
orijinal GGUF dosyaları değişmez. Temel model/adaptör eşleşmezse yükleme
hatasını kontrol et. Adaptör performansını ayrı, eğitimde kullanılmamış test
kümesinde ölçmeden kullanma. Eğitim verisi ve model lisansını kontrol et; özel
verileri depoya commit etme.

## İsteğe bağlı internet araştırması

Güncel bilgi, haber, fiyat, sürüm veya çevrimiçi doğrulama gerektiren sorularda
uygulama model rolünden bağımsız olarak DuckDuckGo'da otomatik arama yapar;
ayrıca arama kutusu seçmen gerekmez. API'de `web_search` açıkça etkinleştirilirse
her model için arama yapılır. Yalnızca
mesajındaki soru aramaya gönderilir; eklediğin
dosyaların içeriği arama sorgusuna katılmaz. En fazla beş HTTPS arama sonucu
alınır, erişilebilen sayfalardan metin çıkarılır, kaynak parçaları yerel indekse
eklenir ve en fazla dört kaynak model tarafından karşılaştırılıp birleştirilerek
sohbet içinde yanıtlanır. Uygulama seni sonuç sitelerine yönlendirmez; yanıtta
kaynak başlığı ve alan adı gösterilebilir. Açıkça indirme bağlantısı
istemedikçe web adresleri yanıta eklenmez; dosyalar kendiliğinden indirilmez.
Google'da oluşturduğun özel arama motoru ayarları DuckDuckGo'ya aktarılmaz.
Ancak aynı alan adlarını DuckDuckGo'da `site:ornek.com arama terimi` biçiminde
arayabilirsin. Güncel bilgi gerektiren sorguların yanı sıra model sağlanan bilgi
ve bağlamla güvenilir yanıt veremediğini belirttiğinde de otomatik arama yapılır.
Selamlaşma ve yanıtlanabilen günlük sohbetlerde gereksiz web araması yapılmaz.
DuckDuckGo araması için API anahtarı gerekmez.

API'den de çağrılabilir:

```powershell
$body = @{ query = "Docker Compose resmi belgeleri"; max_results = 5 } |
  ConvertTo-Json
Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8000/api/arastir `
  -ContentType "application/json" -Body $body
```

Özel repo kodunu, dosya eklerini, sohbet geçmişini veya gizli anahtarları web
aramasına gönderme. Sayfa indirme yalnızca herkese açık HTTPS adreslerine
yapılır; sayfa boyutu ve kaynak adedi sınırlıdır. Arama/indirme site tarafından
engellenebilir, JavaScript ile üretilen sayfa okunamayabilir. Böyle bir durumda
sağlayıcının kısa açıklaması kullanılabilir veya kaynak atlanır. Kaynak ve
iddiaları doğrula; arama özelliği yanlış bilgiyi ortadan kaldırmaz. Sayfa
metinleri kısa parçalara bölünüp yerel indekste saklanır; yanıtta uzun metinleri
aynen kopyalamak yerine özetleme yönergesi uygulanır.

## Uzman modeller ve görsel üretimi

API, istek içeriğini yerel kurallarla uygun GGUF sohbet modeline yönlendirir;
arayüzde model seçmen gerekmez:

- **Flash:** genel/günlük görevler, `models/qwen3-4b-q4_k_m.gguf`
  (Qwen3-4B Q4_K_M; yaklaşık 2,5 GB).
- **Reasoning:** matematik, fizik, kimya ve algoritmik problem çözme.
- **Coder:** yazılım. Her iki uzman rolü de
  `models/qwen3.5-35b-a3b-q4_k_m.gguf` dosyasındaki Qwen3.5-35B-A3B Q4_K_M
  modelini paylaşır (yaklaşık 22 GB; [Hugging Face deposu](https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF)).
- **Floppy:** görsel oluşturma isteği algılandığında otomatik çalışan, yerel
  Stable Diffusion XL Base 1.0 modeli (`models/stable-diffusion-xl-base-1.0`).

Yalnız bir sohbet modeli aynı anda bellekte tutulur. Otomatik seçim, isteği
yerel anahtar-kelime ve matematik ifadesi kurallarıyla anında yönlendirir;
modelden ayrıca sınıflandırma yanıtı beklemez. Belirsiz istekler doğrudan
Flash'a gider. Flash dosyası yoksa ve başka bir sohbet modeli mevcutsa o model
kullanılır. Yanıtın sonunda seçilen uzman rolü durum alanında
gösterilir. Görsel üretimi sohbet modellerinden bağımsız bir görevdir ve ayrı
görsel modeli olan **Floppy**, sohbette “logo oluştur”, “resim çiz” veya
“görsel üret” gibi bir istek algılandığında otomatik devreye girer. İstek
görsel üretimi değilse metin uzmanlarından biri seçilir. Seçilen uzman modeli
yoksa API dosya yolunu içeren açık bir hata döndürür.
İş sürerken GUI, web kaynaklarını toplama, dosyaları analiz etme, istek türüne
göre yerel model yönlendirme, kod/çözüm hazırlama ve yanıt üretme aşamalarını sohbet içindeki bir
simgeyle ve üst durum alanında canlı gösterir.
Bağlam varsayılanı 2.048, yanıt üst sınırı 512 token'dır. Bağlam sınırı aşılırsa
uygulama eski sohbet turlarını sırayla çıkarıp isteği yeniden dener; mevcut soru
korunur. Kısa ve tekrarsız yanıtlar için modelden önce sonucu vermesi istenir;
daha uzun yanıtı yalnızca kullanıcı isterse üretmesi söylenir. Modellerin varsayılan yolları sırasıyla `INK_AI_FLASH_MODEL_PATH`,
`INK_AI_REASONING_MODEL_PATH` ve `INK_AI_CODER_MODEL_PATH` ile değiştirilebilir.
`INK_AI_GPU_LAYERS` GPU katman sayısını ayarlar. 7 GB üzerindeki özel sohbet
modellerinde varsayılan 12 katman, daha küçük modellerde uygun tüm katmanlar
kullanılır. Qwen3.5 Q4_K_M dosyası yaklaşık 20,5 GiB olduğundan 16 GB RAM ve
8 GB VRAM bulunan bir bilgisayarda yükleme yine de yetersiz bellek hatası
verebilir veya disk takası nedeniyle yavaşlayabilir. GPU belleği yetersizse
GPU katmanını azalt; toplam bellek yetersizse daha küçük bir GGUF seç.
`INK_AI_CONTEXT_SIZE` bağlam boyutunu, `INK_AI_MAX_RESPONSE_TOKENS` azami yanıt
uzunluğunu ayarlar. Daha
uzun bağlam/yanıt daha fazla bellek ve süre kullanır; bu değerler modelin eğitim
bilgisini büyütmez.
Windows'ta metin modelleri llama.cpp'nin Vulkan destekli resmi wheel'iyle
çalışır ve desteklenen katmanlar GPU'ya aktarılır. `/api/saglik` yanıtındaki
`gpu_offload_supported` alanı bunu bildirir; hızlandırma yoksa GUI durum alanı
CPU'da çalıştığı konusunda uyarır. PyTorch'un CUDA'yı görmesi tek başına llama.cpp
GPU desteğinin kurulu olduğu anlamına gelmez.

GGUF modellerini, Stable Diffusion XL Base 1.0'ın FP16 bileşenlerini ve ayrı
görsel güvenlik denetleyicisini indirmek için (yaklaşık 20 GB model verisi; ilk
`uv sync` CUDA PyTorch için ayrıca yaklaşık 2,6 GB paket indirir):

```powershell
uv run --no-sync python -m ink_ai.download_models
```

Görsel üretimi üçüncü taraf bir inference API'sine gönderilmez. FastAPI,
Stable Diffusion XL'i yerelde `models/stable-diffusion-xl-base-1.0` klasöründen
çalıştırır ve PNG'leri `%USERPROFILE%\.ink-ai\generated` içine yazar. SDXL daha
yüksek VRAM gerektirdiğinden küçük GPU'larda sıralı CPU offload kullanılır; üretim
daha yavaş olabilir. Ayrı yerel güvenlik denetleyicisi
`models/stable-diffusion-safety-checker` altında tutulur.
`INK_AI_IMAGE_MODEL_PATH` görsel model dizinini değiştirebilir.
Görsel güvenlik denetiminden geçmeyen üretimler siyah görsel yerine açıklayıcı
bir `422` hatası döndürür. Model ağırlıkları indirme sonrasında yerelden okunur;
görsel istemleri harici inference servislerine gönderilmez.

Yerel modeller de her şeyi bilmez: yanıt doğruluğu ve güncelliği modelin eğitim
verisiyle sınırlıdır. İsteğe bağlı web araştırması doğrulanmamış bilgiyi ortadan
kaldırmaz; kaynakları ayrıca doğrula. API varsayılan olarak yalnızca bu bilgisayardaki
`127.0.0.1:8000` adresinde dinler.

## API sınamaları

Proje ortamında sınamaları çalıştırın:

```powershell
uv sync --group dev
uv run pytest tests\test_api.py
```
# Typer DC

Lokalna aplikacja webowa do analizy meczów piłkarskich modelem **Dixona-Colesa (1997)**.
Pokazuje prawdopodobieństwa wyników, porównuje je z kursami bukmacherów i – przede wszystkim –
**mierzy, czy model w ogóle ma przewagę nad rynkiem**. Nie obiecuje wygranych: w obecnych
backtestach (Premier League i Ekstraklasa) model rynku **nie pokonuje**.

> Typer DC to narzędzie analityczne, nie doradztwo. Graj wyłącznie u legalnych bukmacherów
> (w Polsce: z zezwoleniem Ministra Finansów), tylko od 18 lat i tylko w ramach budżetu,
> którego stratę możesz zaakceptować.

## Instalacja

Wymagany Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Uruchomienie

```bash
streamlit run app.py
```

Aplikacja otworzy się w przeglądarce (domyślnie http://localhost:8501). Przy pierwszym
uruchomieniu pobierze dane z football-data.co.uk (kilkanaście sekund).

Narzędzia wiersza poleceń:

```bash
python cli.py summary E0 POL       # pobierz dane i pokaż podsumowanie per sezon
python cli.py backtest E0 POL      # walk-forward backtest + werdykt (ok. 20 s na ligę)
python cli.py backtest E0 --no-tune --offline
```

Testy:

```bash
pytest
```

## Zakładki aplikacji

1. **Mecze** – nadchodzące mecze wybranej ligi (z plików fixtures), prawdopodobieństwa modelu
   vs rynku, oznaczenie value. Szczegóły meczu: oczekiwane gole, heatmapa wyników, najbardziej
   prawdopodobne wyniki, rynki 1X2 / O/U 1.5, 2.5, 3.5 / BTTS. Można wpisać kursy własnego
   bukmachera – aplikacja policzy value i stawkę (1/4 Kelly). Gdy w pliku fixtures nie ma meczów
   ligi, można wybrać dowolną parę drużyn ręcznie. Na górze baner z wynikiem ostatniego backtestu.
2. **Ranking siły** – atak/obrona każdej drużyny, przewaga gospodarzy, ρ, wykresy.
   Beniaminki i drużyny z małą liczbą meczów są oznaczone ⚠️.
3. **Backtest** – wybór ligi i sezonów, metryki model vs rynek, kalibracja, krzywe bankrolla
   (stawka płaska i Kelly), CLV, wyniki per sezon i per typ, lista zakładów (CSV) i werdykt.
4. **Dziennik** – ręczne dodawanie typów (mecz, rynek, kurs, stawka), automatyczne rozliczanie
   po pobraniu wyników, ROI, trafność, CLV. Miesięczny limit budżetu z `config.yaml`: ostrzeżenie
   przy 80%, a po wyczerpaniu **blokada dodawania** do końca miesiąca (blokada działa też w
   warstwie bazy, nie tylko w UI).

## Struktura projektu

| Plik | Zawartość |
|---|---|
| `config.yaml` | ligi, parametry modelu, rynku, stawek, backtestu i budżetu |
| `data.py` | pobieranie, cache na dysku, ujednolicanie kolumn i nazw drużyn |
| `model.py` | model Dixona-Colesa, macierz wyników, rynki, symulacja danych |
| `market.py` | usuwanie marży (proporcjonalnie, Shin), value, Kelly |
| `backtest.py` | walk-forward, strojenie ξ, metryki, symulacja zakładów, CLV, werdykt |
| `journal.py` | dziennik typów w SQLite, budżet, rozliczanie |
| `charts.py` | wykresy Plotly (paleta sprawdzona pod kątem daltonizmu, osobne odcienie trybu ciemnego) |
| `app.py` | interfejs Streamlit |
| `cli.py` | podsumowanie danych i backtest z wiersza poleceń |
| `tests/` | testy pytest |

## Dane

Źródło: [football-data.co.uk](https://www.football-data.co.uk).

- **Ligi główne** (np. `E0` Premier League): plik per sezon `mmz4281/<sezon>/<kod>.csv`.
  Kursy przedmeczowe (`AvgH/D/A`) i zamknięcia (`AvgCH/D/A`), także over/under 2.5.
- **Extra leagues** (np. `POL` Ekstraklasa): jeden plik zbiorczy `new/POL.csv`.
  **Zawiera wyłącznie kursy zamknięcia** (`PSCH`, `MaxCH`, `AvgCH`, `BFECH`, `B365CH`), więc
  dla Ekstraklasy `odds_* = closing_odds_*`, zakłady w backteście są symulowane po kursach
  zamknięcia, a CLV w backteście jest niedostępne (w dzienniku – dostępne, bo porównujemy Twój kurs
  z kursem zamknięcia z pliku).
- **Nadchodzące mecze**: `fixtures.csv` (ligi główne) i `new_league_fixtures.csv` (extra).
  Pliki obejmują zwykle tylko najbliższe dni – poza tym oknem lista bywa pusta.

Format ujednolicony: `date, time, league, season, home, away, hg, ag, odds_h/d/a,
closing_odds_h/d/a, odds_over25/under25, closing_odds_over25/under25, odds_source,
closing_source, odds_are_closing`.

Uwagi:
- Źródło kursów wybierane jest wg listy `odds_priority` / `closing_priority` (domyślnie średnia
  rynkowa `Avg`, potem Pinnacle, potem Bet365). Średnia jest pierwsza, bo jest dostępna we
  wszystkich sezonach – **kolumny Pinnacle zniknęły z plików od sezonu 2026/27** (a w POL.csv
  w trakcie 2025/26).
- Nazwy drużyn są ujednolicane (ogonki, białe znaki, aliasy, np. `Gornik Z.` → `Gornik Zabrze`,
  `Ruch` → `Ruch Chorzow`). Własne aliasy: `data.team_aliases` w `config.yaml`.
- Cache: `data/cache/`. Zakończone sezony nie są pobierane ponownie; pliki bieżącego sezonu,
  zbiorcze i fixtures – gdy są starsze niż `max_age_hours` (domyślnie 12 h). Bez internetu
  aplikacja używa kopii z cache.

## Model

Dla meczu gospodarze *i* – goście *j*:

```
log λ = μ + γ + att_i − def_j        (oczekiwane gole gospodarzy)
log ν = μ + att_j − def_i            (oczekiwane gole gości)
P(x, y) = τ(x, y) · Pois(x; λ) · Pois(y; ν)
```

- `att` – atak, `def` – obrona (więcej = lepiej), `γ` – przewaga gospodarzy, `μ` – poziom bramek.
  Identyfikowalność: `Σ att = 0`, `Σ def = 0`.
- Korekta Dixona-Colesa dla niskich wyników: `τ(0,0) = 1 − λνρ`, `τ(0,1) = 1 + λρ`,
  `τ(1,0) = 1 + νρ`, `τ(1,1) = 1 − ρ`, w pozostałych przypadkach 1.
- **Waga czasowa** meczu: `exp(−ξ · dni_od_meczu)`. Domyślnie ξ = 0.0019/dzień (waga 0.5 po
  roku); **strojenie ξ** wybiera wartość z siatki minimalizującą log-loss 1X2 w walk-forward na
  sezonach walidacyjnych (dwóch sezonach tuż przed testem).
- Estymacja: maksymalizacja ważonej log-wiarygodności (`scipy.optimize`, L-BFGS-B, analityczny
  gradient – dopasowanie trwa ~35 ms). Słaba kara L2 (`ridge`) stabilizuje skrajne wartości.
- Okno treningowe: ostatnie `train_days` dni (domyślnie 3 lata).
- Wynik: macierz prawdopodobieństw 0..10 goli → 1X2, over/under 1.5/2.5/3.5, BTTS, najbardziej
  prawdopodobne wyniki.
- **Beniaminki / drużyny bez historii**: drużyna bez meczów w oknie dostaje parametry „fallback”
  = średnia `fallback_bottom_k` (3) najsłabszych drużyn. Drużyna z mniej niż `min_matches` (6)
  meczami jest ściągana w stronę fallbacku przez prior w funkcji wiarygodności, którego siła maleje
  do zera wraz z kolejnymi meczami. W UI takie drużyny są oznaczone ⚠️.

## Rynek

- Prawdopodobieństwa implikowane `1/kurs` zawierają marżę. Usuwanie marży:
  **proporcjonalnie** (`p_i = (1/o_i) / Σ 1/o_j`) oraz **metodą Shina** (uwzględnia efekt
  faworyt–outsider; parametr z wyznaczany tak, by prawdopodobieństwa sumowały się do 1).
- **Value** = `p_model · kurs − 1`. Typ jest value powyżej progu (domyślnie 5%).
- **Stawka**: płaska (`flat_stake`) lub **ułamkowy Kelly**: `bankroll · min(¼ · f*, 2%)`,
  gdzie `f* = (p·o − 1)/(o − 1)`.

## Backtest

- **Walk-forward**: mecze testowe są dzielone na bloki co ≥ 3 dni (kolejka weekendowa
  i środkowa tygodnia osobno). Dla każdego bloku model jest trenowany **wyłącznie na meczach
  sprzed daty bloku**. Kod twardo sprawdza ten warunek, a testy weryfikują brak wycieku na trzy
  sposoby (podgląd danych widzianych przez model, „zatrucie” przyszłych wyników, strojenie ξ
  tylko na danych sprzed testu).
- Metryki vs rynek bez marży: **log-loss**, **Brier**, RPS, wykres **kalibracji**; różnica
  log-loss z 95% przedziałem ufności (bootstrap sparowany).
- Symulacja zakładów value (najwyżej jeden typ na mecz, kursy 1.2–10): liczba zakładów, ROI,
  **95% przedział ufności ROI (bootstrap)**, maksymalne obsunięcie, bankroll przy ¼ Kelly.
- **CLV**: kurs przy typie vs kurs zamknięcia oraz wartość oczekiwana względem zamknięcia bez marży.
- **Werdykt** – wprost, także gdy wynik jest zły. „Pokonuje rynek” tylko wtedy, gdy log-loss jest
  istotnie lepszy **i** dolna granica przedziału ROI jest powyżej zera.

### Wyniki (stan na 22.10.2026, sezony testowe 2024/25–2026/27)

| | Premier League | Ekstraklasa |
|---|---|---|
| Mecze testowe | 830 | 709 |
| Log-loss 1X2: model / rynek | 1.0055 / 0.9943 | 1.0610 / 1.0383 |
| Różnica (95% CI) | +0.011 [−0.001, +0.023] | +0.023 [+0.008, +0.038] |
| Zakłady value / ROI (95% CI) | 535 / −5.9% [−19.4%, +8.9%] | 398 / −8.8% [−22.1%, +5.4%] |
| CLV | ruch kursu ≈ 0%, EV vs zamknięcie −6.0% | niedostępne (tylko kursy zamknięcia) |
| **Werdykt** | **model NIE pokonuje rynku** | **model NIE pokonuje rynku** |

Wniosek nie zależy od ξ (sprawdzone dla 0, 0.0019 i 0.0035). Over/under 2.5 w Premier League:
również gorzej od rynku (log-loss +0.014, istotnie).

## Konfiguracja (`config.yaml`)

Najważniejsze pola: `leagues` (dodanie ligi głównej to 3 linijki: nazwa, `type: main`, kod
z football-data), `model.xi`, `model.xi_grid`, `model.train_days`, `market.value_threshold`,
`market.margin_method`, `staking.*` (bankroll, stawka płaska, ułamek Kelly, limit %),
`backtest.validation_seasons`, `budget.monthly_limit`, `journal.db_path`.

## Ograniczenia

- **Model widzi tylko gole.** Rynek uwzględnia składy, kontuzje, transfery, xG, motywację,
  pogodę – dlatego jest trudny do pokonania. „Value” wskazane przez model to najczęściej błąd
  modelu, a nie błąd rynku.
- Przewaga gospodarzy jest jedna dla całej ligi i stała w oknie treningowym.
- Fallback zakłada, że drużyna bez historii jest **beniaminkiem z niższej ligi** (słaba).
  Drużyna spadająca z wyższej ligi (np. w Championship) jest wtedy niedoszacowana.
  Po `min_matches` meczach prior znika – beniaminek z dobrym startem może być przeszacowany.
- Backtest gra po **średnich kursach rynkowych** (dla Ekstraklasy – po kursach zamknięcia).
  Marża średniej w PL to 4.5–5.8% (Pinnacle ~3.6%, najlepsza cena 1.0–2.3% w sezonach
  2024/25–2025/26), więc gra u bukmachera z niską marżą poprawiłaby ROI o 1–2 pp – za mało, by
  zmienić werdykt. Kursy maksymalne z plików są nierealistyczne (pojedyncze, szybko korygowane).
- Przedziały ufności ROI są szerokie nawet przy kilkuset zakładach – sama dodatnia stopa zwrotu
  na krótkim okresie niczego nie dowodzi. CLV i log-loss to szybsze i pewniejsze sygnały.
- Bootstrap zakłada niezależność zakładów (w jednej kolejce mogą być skorelowane).
- Pliki football-data są aktualizowane z opóźnieniem (wyniki zwykle w ciągu 1–2 dni,
  fixtures na kilka dni przed kolejką).

## Odpowiedzialna gra

Ustaw limit w `budget.monthly_limit` i go nie podnoś „na odegranie się”. Zakłady to rozrywka
z ujemną wartością oczekiwaną – ten projekt pokazuje to liczbami.

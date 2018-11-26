# Konvoluční neuronové sítě
Konvoluční neuronové sítě jsou odnoží neuronových sítí vytvořených pro efektivnější práci s obrázky. Zavádí několik nových prvků, nejdůležitější z nich je samotná konvoluční vrstva obvykle následovaná *pooling* vrstvou.
### Konvoluce
[Konvoluční vrstva](https://en.wikipedia.org/wiki/Convolutional_neural_network) slouží k extrahování pro NN důležitých částí obrázku. V podstatě se jedná o jednoduchý filter (tréninkem adaptovatelný), který projde každý pixel obrázku (a jemu příslušící okolí), spočítá skalární součin s pokrytým objemem a toto číslo následně prezentuje jako jeden pixel transformovaného obrázku.

<img src=./images/convolution_schematic.gif alt="Konvoluční vrstva" style="height: 250px;\"/>

Jelikož je filtr (oranžová část obrázku) schopný učení, tj. může měnit své hodnoty, má NN možnost si sama vybrat, které části obrázku ji zajímají.

Použitím více filtrů v jedné vrstvě obvylke zmenšujeme výšku a šířku "obrázku", za to zvětšujeme hloubky obrázku.

<img src=./images/Conv_layers.png alt="Konvoluční architektura - schématicky" style="height: 250px;\"/>

### Pooling
Pooling vrstvy se obvykle vkládají vždy mezi konvoluční vrstvy, aby zmenšily rozměry transformovaných obrázků, které prochází neuronovou sítí. Jelikož obecně *pooling* není trénovatelný, pomáhá nám zmenšováním počtu parametrů proti *overfitingu*.

<img src=./images/pool.jpeg alt="Downsampling funkce poolingu." style="height: 250px;\"/>

Ukazuje se, že nejúčinější je tzv. *max pooling*, který také prochází obrázek jako filtr, ale místo toho, aby prováděl skalární součin, vždy vybere pixel s maximální hodnotou a ten použije jako jeden pixel v transformované reprezentaci obrázku.

<img src=./images/maxpool.jpeg alt="Maxpooling" style="height: 250px;\"/>

Více se o konvolučních sítích obecně lze dočíst například [zde](https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/).

## Modely
### 01. Jednoduchá CNN
V prvním případě implementujeme jednoduchou [konvoluční neuronovou síť](https://en.wikipedia.org/wiki/Convolutional_neural_network) s jednou [konvoluční vrstvou](https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/) a dvěma [plně propojenými vrstvami](https://www.reddit.com/r/MachineLearning/comments/3yy7ko/what_is_the_difference_between_a_fullyconnected/).

### 02. Hlubší CNN
V druhém přípaně implementujeme o něco komplexnější model s více konvolucemi, přidáme také  [*'pooling'* vrstvy](https://www.quora.com/What-is-pooling-in-a-deep-architecture), konkrétně [*' maximum pooling'* vrstvy](https://www.quora.com/What-is-max-pooling-in-convolutional-neural-networks).

## Data
Jako zdroj dat nám poslouží [Kaggle](https://www.kaggle.com),  konkrétně jedna z jeho známých [výukových soutěží](https://www.kaggle.com/c/digit-recognizer) na strojové vidění. Data stáhnete také [zde](https://www.kaggle.com/c/digit-recognizer/download/train.csv), mají přibližnš 78 MB. Jedná se o 28x28px obrázky uložené ve formátu .csv. Má dohromady 785 sloupců, kde první číslo v řádku je hledaná číslice a následující čísla jsou příslušné pixely definující obrázek. Každý pixel je představen jedním číslem mezi 0 a 255, jedná se tedy o dvoubarevné obrázky. Další informace o datech lze dohledat na  [stránce jim věnovené](https://www.kaggle.com/c/digit-recognizer/data) nebo ve složce s daty.

## Úspěšnost
| Model                | Train Accuracy | Validation Accuracy | Test Accuracy |
|----------------------|:--------------:|:-------------------:|:-------------:|
| 1. CNN               |         94.5 % |              96.1 % |        95.9 % |
| 2. CNN (400 steps)   |         97.7 % |              97.7 % |        97.1 % |
| 2. CNN (500 steps)   |         96.9 % |              98.2 % |        98.0 % |

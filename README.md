# Fakefilm
A neural networks classifier for fake film

There are many bad film in company's storage. For example:

bad film images
![1](pic/bad.jpg?raw=true "1")
![2](pic/bad.jpg?raw=true "2")
![3](pic/bad.jpg?raw=true "3")

good film images
![1](pic/good.jpg?raw=true "1")
![2](pic/good.jpg?raw=true "2")
![3](pic/good.jpg?raw=true "3")

In order to find them, I made this classifier.

## Dependents
* python3
* tensorflow
* opencv3

## Test
B_XXXXX is name of bad film.
G_XXXXX is name of good film.

FilmName(incomplete) | Classifications/Total | Accuracy |
--------------------------------------------------------|
All Images           | 152164/158996        | 95.700000  |
Bad Film Images      | 31041/31822          | 97.550000  |
Good Film Images     | 121123/127174        | 95.240000  |
B_[西游              | 5035/5060            | 99.510000  |
B_生化               | 3495/3907            | 89.450000  |
B_[血战              | 2571/2648            | 97.090000  |
B_西游               | 5709/5872            | 97.220000  |
B_[BT下              | 1570/1618            | 97.030000  |
B_[迅雷              | 5722/5741            | 99.670000  |
B_长城[              | 6939/6976            | 99.470000  |
G_湄公               | 7134/7134            | 100.000000 |
G_www.a              | 7240/7252            | 99.830000  |
G_西部               | 2054/2064            | 99.520000  |
G_[Skyt              | 1237/1387            | 89.190000  |
G_天海               | 7914/9080            | 87.160000  |
G_【ye3              | 2173/2481            | 87.590000  |
G_看中               | 4397/5255            | 83.670000  |
G_麻                 | 10183/10876          | 93.630000  |
G_虹猫仗             | 997/1022             | 97.550000  |
G_女生               | 5361/6166            | 86.940000  |
G_[名侦              | 1262/1492            | 84.580000  |
G_diogu              | 5749/5873            | 97.890000  |
G_MIAD6              | 9387/9388            | 99.990000  |
G_17_韓              | 3652/4137            | 88.280000  |
G_04.wm              | 2770/2778            | 99.710000  |
G_生化               | 5821/5821            | 100.000000 |
G_金瓶               | 6519/6684            | 97.530000  |
G_Kung               | 4730/4730            | 100.000000 |
G_C客                | 6382/6382            | 100.000000 |
G_你de               | 6422/6426            | 99.940000  |
G_【每               | 4397/4612            | 95.340000  |
G_20_                | 1045/1045            | 100.000000 |
G_极品美             | 532/532              | 100.000000 |
G_                   | 13765/14557          | 94.560000  |

## Usage
You can retrain by youself or use my model.


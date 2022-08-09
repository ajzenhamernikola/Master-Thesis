# GNNSAT

> **Primena grafovskih neuronskih mreža na predviđanje vremena izvršavanja SAT rešavača**

Kratak opis rada: Cilj ovog rada je predviđanje vremena izvršavanja SAT rešavača korišćenjem grafovskih neuronskih mreža i evaluacija pogodnosti tog pristupa za konstrukciju portfolija SAT rešavača. Grafovske neuronske mreže rade direktno nad grafovskim reprezentacijama formula iskazne logike, umesto nad specifično definisanim skalarnim atributima. Kako je već poznato da neuronske mreže koje rade nad izvornim reprezentacijama podataka (što za iskazne formule mogu biti grafovi) često po performansama prestižu sisteme zasnovane na atributima koje su definisali ljudi, postoji osnov da se očekuju bolji rezultati od do sada postignutih. S druge strane, ova vrsta neuronskih mreža može biti računski zahtevnija za primenu. Otud je potrebno evaluirati njihov potencijal za realnu primenu. Uspeh odabranih metoda biće upoređenjen sa klasifikatorima koji su do sada demonstrirali veliki uspeh: _k-najbližih suseda_ i _šume nasumičnih stabala_. Eksperimenti će biti izvršeni korišćenjem programskih jezika Python i C++. Podaci će biti odabrani iz korpusa sa takmičenja SAT Competition.

Rad: [ovde](http://www.racunarstvo.matf.bg.ac.rs/MasterRadovi/2020_06_02_NikolaAjzenhamer/rad.pdf)

---

> **SAT solvers' running time prediction using graph neural networks**

Abstract: This thesis discusses the usage of graph neural networks (GNN) for SAT solvers' running time prediction, with the emplasis in using such an approach in constructing a SAT portfolio. The idea is to use GNN to operate over graph representation of propositional calculus formulae instead of on human-crafted scalar data. As it's known, neural networks operate over raw data representations (which can be graphs in propositional calculus formulae's case) and provide better results over systems that are applied on attributes crafted by humans. Thus, it is plausible to believe that this approach would lead to better results than what had been achieved so far. On the other hand, GNNs can be difficult to use in practical cases, due to the very expensive computational complexity nature. Because of this, it is important to evaluate the practicality of such a system. To test this idea, we've chosen three popular GNN architectures: _graph convolutional network_ (GCN), _graph attention network_ (GAT), and _deep graph convolutional neural network_ (DGCNN). The success of the chosen methods will be compared to classifiers which had proven their success in the past: _k-nearest neighbours_ and _random forests_. The experiments will be achieved using programming languages Python and C++. The data will be selected from the corpus of SAT competition.

Thesis (in Serbian): [here](http://www.racunarstvo.matf.bg.ac.rs/MasterRadovi/2020_06_02_NikolaAjzenhamer/rad.pdf)

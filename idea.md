Je souhaiterais écrire une librairie de tenseurs en C++ équivalent à https://github.com/xtensor-stack/xtensor/ qui permette de faire l'essentiel de ce que sait déjà faire NumPy mais avec des changements de taille. Je souhaites que cette librairie soit intrinsèquement parallèle en s'appuyant fortement sur kokkos (SIMD, mulii GPU, multi CPU). La lazy evaluation est très bien mais il y a potentiellement des grosses pertes de performances. Par exemple, si je fais

c = a + b
d = a + 2*b

je n'ai pas envie de dérouler deux fois la boucle. J'aimerais qu'elle ne soit déclenchée qu'une fois puisque c'est deux fois la même boucle. Il faudrait donc probablement réfléchir à un graphe de taches efficace.
Une vigilance toute particulière devra être faite dans le cas du broadcasting car c'est un vaste sujet qui n'est pas facile à résoudre.
Enfin, les templates permettent de faire beaucoup de choses mais malheureusement les temps de compilation de xtensor sont déraisonnables lorsque les xfunction sont très complexes. Je ne sais pas s'il y a un moyen efficace aujourd'hui de palier à ça. Je crois que les modules dans la librairie standard du C++ ne sont pas encore très fiables.
Bien évidemment, je veux que cette librairie soit multi OS et que l'on puisse l'appeler dans d'autres langages de programmation (Python, Julia, Rust ?). Et à terme, j'aimerais ajouter de l'algèbre linéaire comparable à ce qu'il y a dans PETSc. Le coeur de la librairie devra donc être pensé dès le début pour intégrer tout ça. Le plus important est les performances accrues sur machines HPC distribuées.
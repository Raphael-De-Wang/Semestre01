    set title "Nombre de proies et de prédateurs"
    set xlabel "Nombre d’Itérations"
    set ylabel "Nombre d’animaux"
    plot "myresultat" using 1:2 title "Proies"
    replot "myresultat" using 1:3 title "Prédateurs"


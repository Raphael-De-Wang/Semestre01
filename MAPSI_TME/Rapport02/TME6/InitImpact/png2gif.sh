#!/bin/bash

convert convert -delay 80           \
'evaluation_qualitative(d=10).png'  \
'evaluation_qualitative(d=11).png'  \
'evaluation_qualitative(d=12).png'  \
'evaluation_qualitative(d=13).png'  \
'evaluation_qualitative(d=14).png'  \
'evaluation_qualitative(d=15).png'  \ 
'evaluation_qualitative(d=16).png'  \
'evaluation_qualitative(d=17).png'  \ 
'evaluation_qualitative(d=18).png'  \
'evaluation_qualitative(d=19).png'  \
-loop 0 animated.gif

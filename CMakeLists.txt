project( DetectBarcode )
find_package( OpenCV REQUIRED )
add_executable( DetectBarcode DetectBarcode.cpp DisjointSets.cpp )
target_link_libraries( DetectBarcode ${OpenCV_LIBS} )

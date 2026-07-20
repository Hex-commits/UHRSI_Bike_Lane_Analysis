1. Check, whether I can backtrack to the road classification with the different labels e.g. sidewalk and so on.
2. Remove Shadow Areas from the road-network.
3. Add a 3m buffer, remove a 3m buffer to eliminate outliers from this roadnetwork.


3. Connect Bikelanes on a raster?
4. Remove very split and far-away outliers as false positives.
5. Try to connect them as a "lane", which means that most bikelanes can be connected as a line. So find a mean for the width at the starting and end point, then connect them and narrow it done.



6. Detect Width between road and bikelane
2. Compare it with Unfallatlas-data
3. Visualize it.

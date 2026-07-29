# Individual Report: Bike Lane Analysis on Ultra High Resolution Imagery

Jan-Philipp Wegmeyer

July 29, 2026

## 1 Individual Report

This project fitted my personal development goals well, as I had worked with satellite data before but never with aerial imagery at this resolution. Understanding what 10 cm ground sampling actually gives you, and where it actually stops helping, was the most interesting part for me. It is a very different problem than working with Sentinel data, because suddenly the limiting factor is not the resolution but the road or lane design itself.

I brainstormed the initial idea, came up with the python-based solution architectures and searched for the relevant literature before we started the work itself. I also presented the topic as a speaker, as my understanding of the project is quite deep. On the organisational side I spent a fair amount of effort pushing people to share their results early and to actually incorporate the results of the others into the respective other solutions.

On the technical side I built the pipeline referenced in the GitHub repository, and focused on maintaining a common level of understanding across the three different solutions.
The preprocessing stage fetches and caches the OpenStreetMap geometry, masks the imagery down to a buffer around the road and bike lane network, detects shadows and either corrects or removes them, and boosts the reddish paint so it separates from grey asphalt. These steps are mainly morphological pre-processing, that I assumed would benefit the solution. For the detection stage I first tried a YOLO segmentation model, which did not work well and was later retired, and then moved to a texture embedding approach with a TorchGeo Swin backbone.
However, the limitation of the CNN-Structure failing to recognize the sharpness of the edges that would work for this use-case.  On top of that I added the edge tracing, the shape regularisation, the bridging of gaps between fragments, the road surface reconstruction with an OpenStreetMap width fallback, and finally the measurement of the distance between the road and the bike lane beside it, which was a deliverable of the project.

I also moved every tunable value into one central configuration file and wrote a walkthrough so the others could see each stage on one fixed example, which makes this project entirely reproducible.

The group work went well overall, but the time management was not planned greatly. Information exchange was the main problem, because at any given moment everyone was at a different state of progress, so a result that was ready on one side had nothing to compare to on the other. It was also hard to find real common ground for working together with the the different solution techniques (Python, ArcGis, QGis) we had.

If I did this again I would fix the exchange problem first, maybe setup a common database and semantics to store the respective results. However, the time for this project at the end of the semester was also limited to the other projects in the past semester. The technical side I would keep almost as it is, maybe drop the CNN, but still the findings were quite telling. The open question is the detection quality, since the coarse output of this Python-backbone is still the weakest link, and that is where I would spend the next round of effort or choose an entirely different approach.

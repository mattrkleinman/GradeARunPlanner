# Find A Hilly Run

Python app for finding running/walking routes with specific grades, so you can plan how hilly you want your exercise to be. 

Provide a starting address, total run distance, and outbound grade, and Find A Hilly Run will find routes and score how close they are to your preferences. Routes with a large degree of overlap will be removed, ensuring meaningfully distinct potential routes will be found. 

Routes emphasize consistent outbound movement, so you won't be making lots of small turns or loops. The start and end address are set to be the same, meaning each run will involve turning around at the furthest extent and retracing your steps. This also means while the outbound half of the run will match the specified grade, the grade of the full route will be zero. 

For the first run, after entering the starting location, distance, and grade, use the buttons to:
  (1) retrieve the location of the starting address
  (2) generate a graph file
  (3) upload it in the graph upload box
  (4) generate a routes file
  (5) upload it in the routes upload box
  (6) click the evaluate routes button
  
After this, the desired distance and grade parameters can be changed and the new best matching routes will be plotted after clicking evaluate. Initial map and routes generation can take awhile, especially for distances >20 miles.

For the smoothest operation, generate a unique map and routes file for a given starting location and distance. Unexpected errors can occur when trying to use map and routes files generated with different distance radii from the starting address, due to non-overlapping nodes and routes. Once a map is generated, the desired distance can be adjusted to suit shorter length runs, and matching routes re-evaluated.

Premade demo files can be used to plan running routes in El Cerrito, California. Use the starting address of the City Hall of El Cerrito, "10890 San Pablo Ave, El Cerrito, California, USA". Set the distance to anything 10 miles or less and the grade to any value, with realistic grades between 0 and 5%. In the appropriate upload boxes, upload the graph map file, "10890_San_Pablo_Ave_El_Cerrito_California_USA_10.0_miles.graphml", and the routes file, "10890_San_Pablo_Ave_El_Cerrito_California_USA_10.0_miles.csv". Click "Evaluate routes with current distance and grade inputs." 

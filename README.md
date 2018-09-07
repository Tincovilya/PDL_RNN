# PDL_RNN
Pressure Data Logger Code Documentation Guide
The goal of this guide will be to document which modules/files are responsible for which actions. I will also try to name my files in such a way that it makes sense where they belong and what you need to make them function.
There are two goals behind this project, the first is to take in raw sensor data – combine it – and then produce some kind of graph where a coordinator would be able to look for inferences and issues to plan their next run. The second goal is to produce a recurrent neural network that can detect weld hits and from that calculate an instant velocity. The idea behind the program is outline in the following flow chart.
 
# Preprocessing the Data
For a small run (3 hours) approximately 380 MB of data were generated, this means that pulling in that much data into RAM on a small computer and really doing ANYTHING with python is going to leave you out of RAM very quickly. Think about for a typical run (average of 12 hours) you’ll be over a GB, making a couple of arrays will put you up to 4GB very fast. Then doing operations just the temporary arrays will kill you.
The solution is to first pre-process the data into a database object that can be operated on by Python without pulling it and storing it in memory. I’ve chosen to use h5 to do that, it is a free and widely supported solution that works with the Pandas and NumPy libraries. I’ve written two modules that do this.
 
# PreProcess Data
Inputs
1)	Path for where to find the CSV files for both pressure and acceleration data. EVERYTHING should be in one folder, although obviously you don’t need to be in that same folder for this to work. Just enter that in.
2)	File type – the library I used (glob) can work with different file types. CSV is the easiest thing to use.
Outputs
1)	H5 Data store with one table for temperature/pressure and one table for gyro/acceleration data. Tables called “Temp” and “Gyro”. Called “Sensor_Data.h5”
Details
This module works by making a list of all of the CSV files (assumed to be named as they are now when exported from Pig View, if they start being called something else in a future release would need to change that) also need to make sure that the AGM list is in there.
AGM	Actual Chainage (metres)	Elevation	Date	Time	Type
Launch	0	453.42	2017-12-14	11:27:00 AM	VALVE

First it takes the launch and receive time and can narrow down our data out of the PDL from there. So it is important that those times are accurate.
Goes through all of the CSVs, writes them into an h5 one CSV at a time and then deleting that from memory. This means AT MOST the program would be used a few hundred MB of RAM. Converts the times into a format we want, AND removes any times outside launch/receive. Again need to make sure not NEW data is exported from Pig View or else the hard coded headers will fail.
Once we’ve read in 5 million rows it writes to the storage.
Gather Files
This could probably be renamed just because it is a bit misleading right now, what this actually does is make a combined table in the sensor data h5, so maybe just change the name to combined.
Inputs
1)	Nothing, just need to execute the file in the correct place (Sensor data needs to be in there).
Outputs
1)	Nothing! Makes a new table in “Sensor_Data.h5” that has all data in one table.
This one is needed because the sensors record data at different intervals, so when you put the data together nothing will ever line up and you end up with a ton of NAs or NONE data types.
To alleviate this what I’ve done is go through the data in both tables 5 million rows at a time and append the two dataframes. Some tricky stuff has to occur here because need to ensure everything stays, so used an outer join (pandas), deleted the index and reindexed the whole thing. To get around the problem of missing data I’ve just used the interpolation function, which doesn’t care how much data is missing it will do a linear interpolation. This should be fine because usually the gap is 1-2ms, I honestly don’t care about a change from 1ms to 2ms long. Nothing will be that absurd and it will allow my programs to look down to the ms on EVERYTHING.
Only kind of cool thing I’ve done is if you have a short run it will just read in all data at once and combine it.

# Reccurent Network

Build_Examples
This has too much stuff in it, and I am going to split things out into individual modules depending on what it does. Mostly because this stuff is very time consuming to do and this way each one will be obvious what it is going to do.
Insert_Examples
This module takes in weld times that I’VE picked out from the data and makes a list of weld times. Why? Well to train the network I need to have a list of times, then from those times I can build my examples (Xs and Ys) that will be used to train/test my network.
I’ve arbitrarily decided that my timing will be 5000ms (5 seconds) for each example. The idea being that the computer will get to look at a decent amount of time on either side of a weld hit and learn what IS a weld and NOT a weld. Now in the interests of getting a more equal weighting on my data I could do less time per example and that way the 1s and 0s would be more equal. Something to note is that the weld DOES NOT always occur in the middle. Would just train the network to look in the middle.
Inputs
1)	Nothing – just run it in the same folder at a text file with the weld timings.
Outputs
1)	A list of times that looks like this:
[[start_sample, end_sample, start_time, end_time]]
Where the sample is the weld hit, and the start/end time is the actual total time I would be sampling.
Insert_Ones
This will export my labels for the times made in insert examples. All it does is make an array of 0s and 1s, the 1s exist where the weld was being hit.
Inputs
1)	Y – an array of all 0s (code written to take in any size for export to Tensorflow network)
2)	Time_start
3)	Start_sample
4)	End_sample
5)	Ty – this is the ultimate size of the array we need to feed into the network. So if it is full sized this would be 5000 units long. If we needed a shorter one feed in a smaller number…. Based on kernel, stride length, etc. JUST KNOW THAT IF YOU AREN’T USING A CNN YOU DON’T CARE
Outputs
1)	Y – an array of 0s and 1s in the correct size to feed into the network
Get_xs
This module essentially takes in the time data for where to find the information in the h5 database and then builds the numpy array to pass back so we now would have (from insert ones) a Y and a corresponding X built of the data (Pressure, acceleration, gyro).
Nothing too crazy was done, looks for the data in chunks of 5000000 rows, again for memory reasons I do this. Then what I am doing is seeing if the times exist in that chunk, if it does great built the x array. If it doesn’t then look into the next area and check that out. Now it will also drop the index, ensure all of the X data is interpolated down to the individual ms for continuity.
THIS MODULE NEEDS TO BE RE-WRITTEN WITH THE CODE I HAVE AT HOME BECAUSE I KNOW THAT I WROTE IT TO WRITE THIS INFO INTO AN H5 ARRAY SO THAT WHEN I AM LOOKING FOR EXAMPLES IT JUST STRAIGHT UP PULLS OUT OF THAT. THIS TAKES ~ 20 MINUTES BECAUSE IT IS NOT EFFICIENT. THIS WOULDN’T NEED TO BE DONE EVERY TIME JUST FOR TRAINING THE RNN, BUT IT HELPS TO JUST STORE IT AS H5 THEN NEVER DO IT AGAIN.
Inputs
1)	Time_Start
2)	Time_End
Outputs
1)	A single x example from the time_start to time_end as a numpy array.

# RNN
This module is responsible for building the recurrent net. It has 3 functions that do various different things and are varying degrees of importance. I have utilized Keras and Tensor Flow to do all of the things.
Setup
This function needs you to pass in the X, Y, and length of each label (how long is that vector of 0s and 1s). What it does is normalize the data, and pick a random sample of X/Y to turn into the training and test sets.
Inputs
1)	X – the entire X array
2)	Y – the entire Y array
3)	Ty – size of each vector in the Y array
Outputs
1)	Training sets
2)	Testing sets
3)	The masks derived (since it is a random sample every time)

Model_func
This one builds the model! Just adjust it however you want to get the desired architecture.
Inputs
1)	Input shape
2)	Number of Filters
3)	Kernel Size
4)	GRU units
5)	Stride Length
Outputs
1)	Model – this is just the model object that will contain how the network will train and test itself. Just note that it still needs to be compiled! Hasn’t been yet.

# Main
This one you should call from outside this function and pass in the various things needed. So tell this function the X array, Y array, number of filters the CNN would use, kernel size of the CNN, number of units the GRU will use, and lastly the stride length of the sliding windows in the CNN. The training and testing sets will be created, passed into the model, summarized, the optimizer and loss function defined, and then the model will actually be trained.
Inputs
1)	X
2)	Y
3)	Ty
4)	Number of Filters
5)	Kernel Size
6)	GRU Units
7)	Stride Length
Outputs
1)	Model object
2)	Training mask used for this particular training session
3)	Testing mask used for this particular training session

# Delete
Obviously this one needs to be renamed and rewritten – I would suggest just calling it “Main” instead. The module ties a lot of things together and doesn’t have any functions within it. So just from a programming perspective it sucks. Need to do some work on getting it into a package that I can launch as an exe and actually compile.

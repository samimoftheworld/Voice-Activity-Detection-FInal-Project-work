st=open("silencetime.txt","r")
spt=open("speechtime.txt","r")

nooflines = sum(1 for line in open('/home/sam/test.txt')) #lines in the text.txt file
nooflines1 = sum(1 for line in open('/home/sam/silencetime.txt')) #lines in the silencetime.txt file
nooflines2 = sum(1 for line in open('/home/sam/speechtime.txt')) #lines in the speechtime.txt file

a=[]*nooflines1
b=[]*nooflines2

i=1
foreach( ch in st)
	if(isdigit(ch))
		x= 10*x+ ch.digit()
		if(isalpha(ch.next())) 
			a[i]=x/1000000 #to reduce the timing to seconds / divide by whatever required
foreach( ch in spt)
	if(isdigit(ch))
		x= 10*x+ ch.digit()
		if(isalpha(ch.next())) 
			b[i]=x/1000000 #to reduce the timing to seconds / divide by whatever required
	
j=1
for(i=1;i<=nooflines;i++) #the lop to separate the time line 
    for(k=i;k=a[j];k++)
		#move the content of the segment to the text file containing mfcc of the speech mfcc
	i=a[j] #assign the i to the start limit/time segment of non speech 
	j=j+1 #i.e. the end limit of the non speech
	for(k=i;k=a[j-1]+a[j];k++)
		#move the content of the segment to the text file containing mfcc of the non speech mfcc
	i=a[j] #assign the i to the end limit/time segment of the non speech now as then next limit to be chosen
	j=j+1 #now j is the next limit
	if(i==nooflines)
		for(k=j;k<length(a);k++)
			a[k]=a[k]-60 #to reduce the text by 60 sec as our data is separated by 60s time frames
			

	
		


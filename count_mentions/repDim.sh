dims=("10" "25" "50" "100" "200")
rands=("0" "1")
for((r=0;r<${#rands[@]};r++))
do
	for((d=0;d<${#dims[@]};d++))
	do
		th cnn.lua -glove_dir /home/soumyajit/finetweet/models/unsupervised/glove-avg/glove/ -dim ${dims[d]} -rand_data ${rands[r]}
		th rnn.lua -glove_dir /home/soumyajit/finetweet/models/unsupervised/glove-avg/glove/ -dim ${dims[d]} -rand_data ${rands[r]}
		th bi-rnn.lua -glove_dir /home/soumyajit/finetweet/models/unsupervised/glove-avg/glove/ -dim ${dims[d]} -rand_data ${rands[r]}
		th fasttext.lua
	done
done

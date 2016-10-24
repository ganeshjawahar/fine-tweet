lr=("0.05" "0.1" "0.25" "0.5")
bsize=("64" "128" "256")
num_epochs=("5" "10")
dropout_p=("0.1" "0.3" "0.5" "0.8")
for((l=0;l<${#lr[@]};l++))
do
	for((b=0;b<${#bsize[@]};b++))
	do
		for((ne=0;ne<${#num_epochs[@]};ne++))
		do
			for((dp=0;dp<${#dropout_p[@]};dp++))
			do
				th unsup.lua -lr ${lr[l]} -batch_size ${bsize[b]} -num_epochs ${num_epochs[ne]} -dropout_p ${dropout_p[dp]}
			done
		done
	done
done
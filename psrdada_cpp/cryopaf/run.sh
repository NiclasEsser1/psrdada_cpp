#command block that takes time to complete...
#........
# Make sure that no dada buffers with the same key are alive. Destroy them if necessary...
echo "Destroying existing dada buffer.."
dada_db -k dada -d;

echo "Destroying existing dada buffer.."
dada_db -k dadc -d;

# # Creating dada buffer
echo "Creating new dada buffer numa node 0.."
dada_db -k dada -l -p -b 469762048 -n 8

dada_db -k dadc -l -p -b 234881024 -n 8

# dada_dbnull -k dadc -S >/dev/null &
# pid1=$!
#
# /media/scratch/nesser/Projects/psrdada_cpp/build/psrdada_cpp/cryopaf/beamforming --in_key dada --out_key dadc --kind power &
# pid2=$!
#
# dada_junkdb -k dada -r 100.0 -t 20 details/header_dada.txt &
# pid3=$!
#
# # dada_dbmonitor -k dada
#
# read -p "Press enter to stop"
#
# echo "Killing process $pid1"
# kill -9 $pid1
# echo "Killing process $pid2"
# kill -9 $pid2
# echo "Killing process $pid3"
# kill -9 $pid3

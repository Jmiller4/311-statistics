#!/bin/bash

inputfile=${1}
count=0

mkdir -p age-and-sex
mkdir -p race-and-ethnicity

chmod 777 ./age-and-sex/

cd ./race-and-ethnicity
mkdir -p C02003
mkdir -p B03002

cd ..

chmod 777 ./race-and-ethnicity/C02003/
chmod 777 ./race-and-ethnicity/B03002/


while read -r line
do
    count=$(( ${count}+1 ))
    echo "Block Group ${line}"
    curl -I "https://api.censusreporter.org/1.0/data/show/latest?table_ids=B01001&geo_ids=15000US${line}"
    #> ./age-and-sex/${line}-B01001.json
#    until $(curl -# "https://api.censusreporter.org/1.0/data/show/latest?table_ids=C02003&geo_ids=15000US${line}" > ./race-and-ethnicity/C02003/${line}-C02003.json); do
#	printf '.'
#	sleep 1
#    done
#    until $(curl -# "https://api.censusreporter.org/1.0/data/show/latest?table_ids=B03002&geo_ids=15000US${line}" > ./race-and-ethnicity/B03002/${line}-B03002.json); do
#	printf '.'
#	sleep 1
#    done
    echo "count = ${count}"
    sleep 0.5
done < ${inputfile}


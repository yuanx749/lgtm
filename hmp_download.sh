mkdir -p hmp
curl -L -o hmp/hmp2_metadata_2018-08-20.csv https://g-227ca.190ebd.75bc.data.globus.org/ibdmdb/metadata/hmp2_metadata_2018-08-20.csv
curl -L -o hmp/dysbiosis_scores.tsv https://forum.biobakery.org/uploads/short-url/umwfR0kDJ6s5RXHwtIMgLaOEoOI.tsv
curl -L -o hmp/taxonomic_profiles_3.tsv.gz https://g-227ca.190ebd.75bc.data.globus.org/ibdmdb/products/HMP2/MGX/2018-05-04/taxonomic_profiles_3.tsv.gz
gunzip -f hmp/taxonomic_profiles_3.tsv.gz
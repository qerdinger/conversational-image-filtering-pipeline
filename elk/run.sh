docker build -t cifp-elasticsearch .

docker run -d \
  --name my-cifp-elasticsearch \
  -p 9200:9200 \
  -p 9300:9300 \
  cifp-elasticsearch
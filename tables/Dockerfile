FROM alpine:latest

WORKDIR /src

COPY . /src/

RUN apk update && apk add curl

RUN chmod 777 /src/

RUN chmod 777 /src/fetch-tables.sh

RUN chmod 777 /src/cook-county-bg-geoids.txt

USER root

ENTRYPOINT ["sh","/src/fetch-tables.sh","/src/cook-county-bg-geoids.txt"]
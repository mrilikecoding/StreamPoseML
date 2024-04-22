FROM node:18

WORKDIR /usr/src/app/my-app

ADD package.json package.json
ADD package-lock.json package-lock.json
RUN npm install

ADD . ./

EXPOSE 3000
ENTRYPOINT npm start


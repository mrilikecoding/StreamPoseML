FROM node:latest

WORKDIR /usr/src/app/my-app

COPY yarn.lock package.json ./

RUN yarn install

COPY . .

EXPOSE 3000

ENTRYPOINT yarn dev --host


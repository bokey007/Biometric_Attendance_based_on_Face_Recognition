'use strict';
const electron= require('electron')
const fs=require('fs')
let video
function object_btn()
    {
      function reqListener()
      {
        document.getElementById("video").innerHTML=this.responseText;
      }
      xhttp=newHttpRequest()
      xhttp.addEventListener("object_btn",reqListener);
      xhttp.open('GET , POST','http://localhost:5001');
      http.send()
    }
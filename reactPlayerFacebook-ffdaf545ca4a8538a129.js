(window.webpackJsonp=window.webpackJsonp||[]).push([[14],{nQ6K:function(e,t,n){"use strict";Object.defineProperty(t,"__esModule",{value:!0}),t.default=void 0;var r=function(e){if(e&&e.__esModule)return e;if(null===e||"object"!==i(e)&&"function"!=typeof e)return{default:e};var t=a();if(t&&t.has(e))return t.get(e);var n={},r=Object.defineProperty&&Object.getOwnPropertyDescriptor;for(var o in e)if(Object.prototype.hasOwnProperty.call(e,o)){var u=r?Object.getOwnPropertyDescriptor(e,o):null;u&&(u.get||u.set)?Object.defineProperty(n,o,u):n[o]=e[o]}n.default=e,t&&t.set(e,n);return n}(n("q1tI")),o=n("ODpG"),u=n("AEgf");function a(){if("function"!=typeof WeakMap)return null;var e=new WeakMap;return a=function(){return e},e}function i(e){return(i="function"==typeof Symbol&&"symbol"==typeof Symbol.iterator?function(e){return typeof e}:function(e){return e&&"function"==typeof Symbol&&e.constructor===Symbol&&e!==Symbol.prototype?"symbol":typeof e})(e)}function l(e,t){if(!(e instanceof t))throw new TypeError("Cannot call a class as a function")}function c(e,t){for(var n=0;n<t.length;n++){var r=t[n];r.enumerable=r.enumerable||!1,r.configurable=!0,"value"in r&&(r.writable=!0),Object.defineProperty(e,r.key,r)}}function f(e,t){return(f=Object.setPrototypeOf||function(e,t){return e.__proto__=t,e})(e,t)}function s(e){var t=function(){if("undefined"==typeof Reflect||!Reflect.construct)return!1;if(Reflect.construct.sham)return!1;if("function"==typeof Proxy)return!0;try{return Date.prototype.toString.call(Reflect.construct(Date,[],(function(){}))),!0}catch(e){return!1}}();return function(){var n,r=b(e);if(t){var o=b(this).constructor;n=Reflect.construct(r,arguments,o)}else n=r.apply(this,arguments);return p(this,n)}}function p(e,t){return!t||"object"!==i(t)&&"function"!=typeof t?y(e):t}function y(e){if(void 0===e)throw new ReferenceError("this hasn't been initialised - super() hasn't been called");return e}function b(e){return(b=Object.setPrototypeOf?Object.getPrototypeOf:function(e){return e.__proto__||Object.getPrototypeOf(e)})(e)}function d(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}var h="https://connect.facebook.net/en_US/sdk.js",v=function(e){!function(e,t){if("function"!=typeof t&&null!==t)throw new TypeError("Super expression must either be null or a function");e.prototype=Object.create(t&&t.prototype,{constructor:{value:e,writable:!0,configurable:!0}}),t&&f(e,t)}(i,e);var t,n,u,a=s(i);function i(){var e;l(this,i);for(var t=arguments.length,n=new Array(t),r=0;r<t;r++)n[r]=arguments[r];return d(y(e=a.call.apply(a,[this].concat(n))),"callPlayer",o.callPlayer),d(y(e),"playerID",e.props.config.playerId||"".concat("facebook-player-").concat((0,o.randomString)())),d(y(e),"mute",(function(){e.callPlayer("mute")})),d(y(e),"unmute",(function(){e.callPlayer("unmute")})),e}return t=i,(n=[{key:"componentDidMount",value:function(){this.props.onMount&&this.props.onMount(this)}},{key:"load",value:function(e,t){var n=this;t?(0,o.getSDK)(h,"FB","fbAsyncInit").then((function(e){return e.XFBML.parse()})):(0,o.getSDK)(h,"FB","fbAsyncInit").then((function(e){e.init({appId:n.props.config.appId,xfbml:!0,version:n.props.config.version}),e.Event.subscribe("xfbml.render",(function(e){n.props.onLoaded()})),e.Event.subscribe("xfbml.ready",(function(e){"video"===e.type&&e.id===n.playerID&&(n.player=e.instance,n.player.subscribe("startedPlaying",n.props.onPlay),n.player.subscribe("paused",n.props.onPause),n.player.subscribe("finishedPlaying",n.props.onEnded),n.player.subscribe("startedBuffering",n.props.onBuffer),n.player.subscribe("finishedBuffering",n.props.onBufferEnd),n.player.subscribe("error",n.props.onError),n.props.muted||n.callPlayer("unmute"),n.props.onReady(),document.getElementById(n.playerID).querySelector("iframe").style.visibility="visible")}))}))}},{key:"play",value:function(){this.callPlayer("play")}},{key:"pause",value:function(){this.callPlayer("pause")}},{key:"stop",value:function(){}},{key:"seekTo",value:function(e){this.callPlayer("seek",e)}},{key:"setVolume",value:function(e){this.callPlayer("setVolume",e)}},{key:"getDuration",value:function(){return this.callPlayer("getDuration")}},{key:"getCurrentTime",value:function(){return this.callPlayer("getCurrentPosition")}},{key:"getSecondsLoaded",value:function(){return null}},{key:"render",value:function(){return r.default.createElement("div",{style:{width:"100%",height:"100%"},id:this.playerID,className:"fb-video","data-href":this.props.url,"data-autoplay":this.props.playing?"true":"false","data-allowfullscreen":"true","data-controls":this.props.controls?"true":"false"})}}])&&c(t.prototype,n),u&&c(t,u),i}(r.Component);t.default=v,d(v,"displayName","Facebook"),d(v,"canPlay",u.canPlay.facebook),d(v,"loopOnEnded",!0)}}]);
//# sourceMappingURL=reactPlayerFacebook-ffdaf545ca4a8538a129.js.map
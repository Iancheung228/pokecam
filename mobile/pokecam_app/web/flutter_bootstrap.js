{{flutter_js}}
{{flutter_build_config}}

// Unregister any existing service workers so Safari always loads fresh assets.
if ('serviceWorker' in navigator) {
  navigator.serviceWorker.getRegistrations().then(function(registrations) {
    for (var r of registrations) { r.unregister(); }
  });
}

_flutter.loader.load();

const apiCall = async (path, params) => {
  const rawResponse = await fetch(path, {
    method: "POST",
    headers: {
      Accept: "application/json",
      "Content-Type": "application/json",
    },
    body: JSON.stringify(params),
  });
  return await rawResponse.json();
};

window.addEventListener("DOMContentLoaded", () => {
  const mymap = L.map("mapid").setView([55.7522, 37.6156], 10);
  window.mymap = mymap;
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 25,
  }).addTo(mymap);

  document.getElementById("submit").addEventListener("click", async () => {
    const objectType = document.getElementById("object_type_select").value;

    const response = await apiCall("/calculate", { object_type: objectType });

    L.heatLayer(response.heat_layer, {
      radius: 25,
      blur: 10,
      minOpacity: 0.25,
    }).addTo(mymap);

    response.markers.forEach((marker) => {
      L.marker({ lat: marker.lat, lng: marker.lng }, marker.options)
        .bindPopup(marker.popup)
        .addTo(mymap);
    });
  });
});

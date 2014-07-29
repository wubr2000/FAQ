$(function () {
  var $spinner = $('#waitingSpinner');

  $('form.generate-faq').on('submit', function (e) {
    console.log('clicked!');
    $spinner.show();
  })
})
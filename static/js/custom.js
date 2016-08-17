$(function(){
    $('#add_channel').click(function(){
        var channel = $('.channel');
        if (channel.length == 2) {
            alert("at most 2 channel allowed");
            return false;
        }
        var last_channel= channel.last();
        last_channel.after(last_channel.clone(true));          
    })
    $('#remove_channel').click(function(){
        var channel = $('.channel');
        if (channel.length == 1) {
            alert("at least one channel");
            return false;
        }
        channel.last().remove();        
    })
    $('#add_filter').click(function(){
        var filter = $('.filter');
        if (filter.length == 3) {
            alert("at most 3 filter allowed");
            return false;
        }
        var last_filter= filter.last();
        last_filter.after(last_filter.clone(true));          
    })
    $('#remove_filter').click(function(){
        var filter = $('.filter');
        if (filter.length == 1) {
            alert("at least one filter");
            return false;
        }
        filter.last().remove();        
    });

    /*$("form").submit(function(event){
        event.preventDefault();
        //var data = $(this).serialize();
        var jqxhr = $.ajax({
            method: 'POST',
            url: $(this).attr('action'),
            data: new FormData($(this)),
            //dataType: 'json',
            //Options to tell jQuery not to process data or worry about content-type.
            cache: false,
            contentType: false,
            processData: false
        }).done(function(msg){
            alert("ajax : " + msg);
        }).fail(function(xhr, status) {
            alert("failed : " + xhr.status);
        }).always(function(){
            alert("has step in ajax");
        });
    });*/
})
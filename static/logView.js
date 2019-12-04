var app = new Vue({
    el: '#app',
    data: {
        raw: '',
        list: [],
        log: '',
        floor_map: [[]],
        index: 0,
        action: '',
        intervalIds: [],
        rewardSum: 0,
    },
    async mounted() {
        await axios.get('/log/list').then(response => {
            this.list = response.data.list;
            if(this.list.length > 0){
                this.log = this.list[0];
            }
        });
    },
    created() {
        document.onkeydown = () => {
            this.onKeyDown(event.keyCode);
        };
    },
    methods: {
        getList: function(){
            axios.get('/log/list').then(response => {
                this.list = response.data.list;
            });
        },
        getLogData: function(){
            this.index = 0;
            axios.get('/log/'+this.log).then(response => {
                this.raw = response.data;
                this.rewardSum = response.data.rewardSum;
                this.setAgent();
            });
        },
        setAgent: function(){
            this.floor_map = JSON.parse(JSON.stringify(this.raw.cellMap));
            let agent = this.raw.moveLog[this.index].agent;
            let enemies = this.raw.moveLog[this.index].enemies;
            if(this.index >= this.raw.moveLog.length-1){
                if(this.floor_map[agent.y][agent.x] !== 5){
                    this.floor_map[agent.y][agent.x] = 6;
                }else {
                    this.floor_map[agent.y][agent.x] = 3;
                }
            }else{
                this.floor_map[agent.y][agent.x] = 3;
            }
            enemies.forEach(e => {
                if(e.x !== -1 && e.y !== -1){
                    this.floor_map[e.y][e.x] = 4;
                }
            });
            switch(this.raw.moveLog[this.index].action){
                case 0: this.action = 'attack'; break;
                case 1: this.action = 'up'; break;
                case 2: this.action = 'right'; break;
                case 3: this.action = 'down'; break;
                case 4: this.action = 'left'; break;
            }
        },
        prev: function(){
            this.index -= 1;
            if(this.index <= 0){
                this.index = 0;
            }
            this.setAgent();
        },
        next: function(){
            this.index += 1;
            if(this.index > this.raw.moveLog.length-1){
                this.index = this.raw.moveLog.length-1;
            }
            this.setAgent();
        },
        play: function(){
            let intervalId = setInterval(function(){
                if(app.index >= app.raw.moveLog.length - 1){
                    app.index = 0;
                    app.setAgent();
                    return;
                }
                app.next();
            }, 200);
            this.intervalIds.push(intervalId);
        },
        stop: function(){
            clearInterval(this.intervalIds.pop());
        },
        onKeyDown: function(keyCode){
            switch(keyCode){
                case 37:
                    this.prev();
                    break;
                case 39:
                    this.next();
                    break;
            }
        },
    },
    watch: {
        log: function(){
            this.getLogData();
        }
    },
})
DROP DATABASE IF EXISTS jspL3B4u;

CREATE DATABASE jspL3B4u default character set utf8mb4 collate utf8mb4_general_ci;

USE jspL3B4u;

DROP TABLE IF EXISTS `lvyoujingdian`;

CREATE TABLE `lvyoujingdian` (
	`id` bigint NOT NULL AUTO_INCREMENT,
	`addtime` timestamp NOT NULL default CURRENT_TIMESTAMP,
	`jingdianbianhao` varchar(200)  UNIQUE   COMMENT '景点编号',
	`jingdianmingcheng` varchar(200)    COMMENT '景点名称',
	`tupian` varchar(200)    COMMENT '图片',
	`menpiao` varchar(200)    COMMENT '门票',
	`jingdianxiangqing` longtext    COMMENT '景点详情',
	PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COMMENT='旅游景点';

DROP TABLE IF EXISTS `dingdanxinxi`;

CREATE TABLE `dingdanxinxi` (
	`id` bigint NOT NULL AUTO_INCREMENT,
	`addtime` timestamp NOT NULL default CURRENT_TIMESTAMP,
	`dingdanbianhao` varchar(200)  UNIQUE   COMMENT '订单编号',
	`jingdianbianhao` varchar(200)    COMMENT '景点编号',
	`jingdianmingcheng` varchar(200)    COMMENT '景点名称',
	`tupian` varchar(200)    COMMENT '图片',
	`menpiao` varchar(200)    COMMENT '门票',
	`goumaishuliang` varchar(200)    COMMENT '购买数量',
	`zongjiage` varchar(200)    COMMENT '总价格',
	`yonghuzhanghao` varchar(200)    COMMENT '用户账号',
	`yonghuxingming` varchar(200)    COMMENT '用户姓名',
	`yuyueshijian` datetime    COMMENT '预约时间',
	`ispay` varchar(200)   default '未支付' COMMENT '是否支付',
	PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COMMENT='订单信息';

DROP TABLE IF EXISTS `lvyougonglve`;

CREATE TABLE `lvyougonglve` (
	`id` bigint NOT NULL AUTO_INCREMENT,
	`addtime` timestamp NOT NULL default CURRENT_TIMESTAMP,
	`jingdianbianhao` varchar(200)    COMMENT '景点编号',
	`jingdianmingcheng` varchar(200)    COMMENT '景点名称',
	`tupian` varchar(200)    COMMENT '图片',
	`lvyougonglve` longtext    COMMENT '旅游攻略',
	`fabushijian` date    COMMENT '发布时间',
	PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COMMENT='旅游攻略';

DROP TABLE IF EXISTS `yonghu`;

CREATE TABLE `yonghu` (
	`id` bigint NOT NULL AUTO_INCREMENT,
	`addtime` timestamp NOT NULL default CURRENT_TIMESTAMP,
	`yonghuzhanghao` varchar(200) NOT NULL UNIQUE   COMMENT '用户账号',
	`yonghuxingming` varchar(200) NOT NULL   COMMENT '用户姓名',
	`mima` varchar(200)    COMMENT '密码',
	`xingbie` varchar(200)    COMMENT '性别',
	`nianling` varchar(200)    COMMENT '年龄',
	`lianxidianhua` varchar(200)    COMMENT '联系电话',
	`shenfenzheng` varchar(200)    COMMENT '身份证',
	PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COMMENT='用户';

DROP TABLE IF EXISTS `storeup`;

CREATE TABLE `storeup` (
	`id` bigint NOT NULL AUTO_INCREMENT,
	`addtime` timestamp NOT NULL default CURRENT_TIMESTAMP,
	`userid` bigint NOT NULL   COMMENT '用户id',
	`refid` bigint    COMMENT '收藏id',
	`tablename` varchar(200)    COMMENT '表名',
	`name` varchar(200) NOT NULL   COMMENT '收藏名称',
	`picture` varchar(200) NOT NULL   COMMENT '收藏图片',
	PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COMMENT='收藏表';

DROP TABLE IF EXISTS `news`;

CREATE TABLE `news` (
	`id` bigint NOT NULL AUTO_INCREMENT,
	`addtime` timestamp NOT NULL default CURRENT_TIMESTAMP,
	`title` varchar(200) NOT NULL   COMMENT '标题',
	`picture` varchar(200) NOT NULL   COMMENT '图片',
	`content` longtext NOT NULL   COMMENT '内容',
	PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COMMENT='新闻资讯';


DROP TABLE IF EXISTS `config`;

CREATE TABLE `config`(
	`id` bigint NOT NULL AUTO_INCREMENT,
	`name` varchar(100) NOT NULL COMMENT '配置参数名称',
	`value` varchar(100) COMMENT '配置参数值',
	PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COMMENT='配置文件';

insert into config(id,name) values(1,'picture1');
insert into config(id,name) values(2,'picture2');
insert into config(id,name) values(3,'picture3');
insert into config(id,name) values(4,'picture4');
insert into config(id,name) values(5,'picture5');
insert into config(id,name) values(6,'homepage');


DROP TABLE IF EXISTS `users`;

CREATE TABLE `users`(
	`id` bigint NOT NULL AUTO_INCREMENT,
	`username` varchar(100) NOT NULL COMMENT '用户名',
	`password` varchar(100) NOT NULL COMMENT '密码',
	`role` varchar(100) default '管理员' COMMENT '角色',
	`addtime` timestamp NOT NULL default CURRENT_TIMESTAMP COMMENT '新增时间',
	PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COMMENT='用户表';

insert into users(id,username, password) values(1,'abo','abo');

DROP TABLE IF EXISTS `token`;

CREATE TABLE `token`(
	`id` bigint NOT NULL AUTO_INCREMENT,
	`userid` bigint NOT NULL COMMENT '用户id',
	`username` varchar(100) NOT NULL COMMENT '用户名',
	`tablename` varchar(100) COMMENT '表名',
	`role` varchar(100) COMMENT '角色',
	`token` varchar(200) NOT NULL COMMENT '密码',
	`addtime` timestamp NOT NULL default CURRENT_TIMESTAMP COMMENT '新增时间',
	`expiratedtime` timestamp COMMENT '过期时间',
	PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COMMENT='token表';


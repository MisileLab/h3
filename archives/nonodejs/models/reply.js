'use strict';
const {
  Model, Sequelize
} = require('sequelize');
const db = require('./index.js');
const createReplyEntity = (sequelize, DataTypes) => {
  class Reply extends Model {
    /**
     * Helper method for defining associations.
     * This method is not a part of Sequelize lifecycle.
     * The `models/index` file will call this method automatically.
     */
    static associate(models) {
      models.Reply.belongsTo(models.Post, {foreignKey: 'post_id', targetKey: 'id'})
    }
  }
  Reply.init({
    post_id: DataTypes.STRING,
    body: DataTypes.TEXT,
    author: DataTypes.STRING,
    time: DataTypes.DATE,
    password: DataTypes.STRING
  }, {
    sequelize,
    modelName: 'Reply',
  });
  return Reply;
};
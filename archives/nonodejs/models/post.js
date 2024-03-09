'use strict';
const {
  Model, Sequelize
} = require('sequelize');
const db = require('./index.js');
function createPostEntity (sequelize, DataTypes) {
  class Post extends Model {
    /**
     * Helper method for defining associations.
     * This method is not a part of Sequelize lifecycle.
     * The `models/index` file will call this method automatically.
     */
    static associate(models) {
      // define association here
      models.Post.hasMany(models.Reply, {foreignKey: 'post_id', sourceKey: 'id'})
      models.Post.belongsTo(models.User, {foreignKey: 'author', targetKey: 'username'})
    }
  }
  Post.init({
    title: DataTypes.STRING,
    body: DataTypes.TEXT,
    author: DataTypes.STRING,
    time: DataTypes.DATE
  }, {
    sequelize,
    modelName: 'Post',
  });
  return Post;
}

module.exports = createPostEntity(db.sequelize, Sequelize);

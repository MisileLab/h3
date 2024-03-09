// deno-lint-ignore-file
'use strict';
const {
  Model, Sequelize
} = require('sequelize');
const db = require('./index.js')
const createUserEntity = (sequelize, DataTypes) => {
  class User extends Model {
    /**
     * Helper method for defining associations.
     * This method is not a part of Sequelize lifecycle.
     * The `models/index` file will call this method automatically.
     */
    static associate(models) {
      models.User.hasMany(models.Post, {foreignKey: 'author', sourceKey: 'username'})
    }
  }
  User.init({
    email: DataTypes.STRING,
    username: DataTypes.STRING,
    password: DataTypes.STRING
  }, {
    sequelize,
    modelName: 'User',
  });
  return User;
};

module.exports = createUserEntity(db.sequelize, Sequelize);

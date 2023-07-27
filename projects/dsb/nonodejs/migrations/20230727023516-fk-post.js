'use strict';

/** @type {import('sequelize-cli').Migration} */
module.exports = {
  async up (queryInterface, Sequelize) {
    await queryInterface.addColumn("Posts", "author", {
      type: Sequelize.STRING,
    })
    await queryInterface.addConstraint('Posts', {
      fields: ['author'],
      type: 'foreign key',
      name: 'posts_author_fk',
      references: {
        table: 'Users',
        field: 'username'
      },
      onDelete: 'cascade',
      onUpdate: 'cascade'
    })
  },

  async down (queryInterface, Sequelize) {
    /**
     * Add reverting commands here.
     *
     * Example:
     * await queryInterface.dropTable('users');
     */
    await queryInterface.removeColumn("Posts", "author");
  }
};

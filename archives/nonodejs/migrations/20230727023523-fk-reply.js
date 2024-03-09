'use strict';

/** @type {import('sequelize-cli').Migration} */
module.exports = {
  async up (queryInterface, Sequelize) {
    /**
     * Add altering commands here.
     *
     * Example:
     * await queryInterface.createTable('users', { id: Sequelize.INTEGER });
     */
    await queryInterface.addColumn("Replies", "post_id", {
      type: Sequelize.INTEGER,
    });
    await queryInterface.addConstraint("Replies", {
      fields: ["post_id"],
      type: "foreign key",
      name: "replies_post_id_fk",
      references: {
        table: "Posts",
        field: "id",
      },
      onDelete: "cascade",
      onUpdate: "cascade",
    })
  },

  async down (queryInterface, Sequelize) {
    /**
     * Add reverting commands here.
     *
     * Example:
     * await queryInterface.dropTable('users');
     */
    await queryInterface.removeColumn("Replies", "post_id");
  }
};
